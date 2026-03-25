import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, GFnSelector, M_EDM_ode_v3, ANI_absM_Precond_Wrapper_basis_v2

torch.set_default_dtype(torch.float32)

# ===============================================================
# NEW: teacher clustered PCA utils (minimal add)
# ===============================================================
@torch.no_grad()
def convert_Q_to_V(Q_2d: torch.Tensor, res: int):
    """
    Q_2d: [d, r]  where d=res*res (4096), r=1024
    return V: [r, res, res]
    """
    d, r = Q_2d.shape
    assert d == res * res, f"Q has d={d}, expected {res*res}"
    V = Q_2d.T.reshape(r, res, res).contiguous()
    V = V / (V.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)).view(r, 1, 1)
    return V

@torch.no_grad()
def load_teacher_clustered_basis(path: str, device, res: int):
    """
    teacher .pt contains:
      bases['Q'] shape [K, d, r]  (e.g. [60,4096,1024])
      bases['class_to_cluster'] shape [1000]
    We convert to your expected:
      V_by_label: dict[int]->Tensor[r,res,res]
      label2basis: dict[int]->int
    """
    bases = torch.load(path, map_location="cpu")
    assert isinstance(bases, dict), "teacher basis must be a dict"
    assert "Q" in bases and "class_to_cluster" in bases, f"Missing keys. got={list(bases.keys())}"

    Q_all = bases["Q"]  # [K,d,r]
    class_to_cluster = bases["class_to_cluster"]  # [1000]

    V_by_label = {}
    for k in range(Q_all.shape[0]):
        V_by_label[int(k)] = convert_Q_to_V(Q_all[k], res=res).to(device)

    label2basis = {i: int(class_to_cluster[i]) for i in range(len(class_to_cluster))}
    mn, mx = min(label2basis.values()), max(label2basis.values())
    print(f"[TeacherBasis] Loaded clustered PCA: K={len(V_by_label)} clusters, label2basis range [{mn},{mx}]")

    # sanity
    assert mn >= 0 and mx < len(V_by_label), f"class_to_cluster out of range: [{mn},{mx}] vs K={len(V_by_label)}"
    return V_by_label, label2basis

def make_batch_fn(fn_dict, indices, device):
    """
    fn_dict: {label(str) : GFn}
    indices: LongTensor [B]
    return: callable fn(t) that:
      - for scalar t: returns per-sample values shape [B]
      - for vector t (used by invert): falls back to a reference GFn
    """
    ref = next(iter(fn_dict.values()))  # pick any class as reference for inversion/meta

    def batch_fn(t):
        t = torch.as_tensor(t, device=device)

        # IMPORTANT:
        # invert_g_bisect never calls batch_fn(t); it calls batch_fn.g_and_grad(t).
        # But to be safe, if someone calls batch_fn with vector t, we fall back to ref.
        if t.numel() != 1:
            return ref(t)[0], None

        # scalar t -> per-sample schedule
        out = torch.empty(indices.shape[0], device=device, dtype=torch.float32)
        for c in indices.unique():
            c_int = int(c.item())
            mask = (indices == c_int)
            out[mask] = fn_dict[str(c_int)](t)[0]  # scalar -> fill [mask.sum()]
        return out, None

    # ---- make it look like a GFn for invert_g_bisect ----
    batch_fn.times = ref.times
    batch_fn.T     = ref.T
    batch_fn.g0    = ref.g0

    # CRITICAL: inversion wants a single curve; use ref curve for g_and_grad
    batch_fn.g_and_grad = ref.g_and_grad

    return batch_fn

# ===============================================================
# Helper: load model checkpoint (g-ani / g-h / g-iso)
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device, args=None):
    basis_dir = os.path.join(ckpt_root, dataset)

    V_by_label_path  = os.path.join(basis_dir, "basis_by_label.pt")
    V_global_path    = os.path.join(basis_dir, "basis_global.pt")
    label2basis_path = os.path.join(basis_dir, "label2basis.pkl")

    V_by_label = None
    V_global = None
    label2basis = None

    # NEW: infer res here for teacher basis conversion (keep local & minimal)
    if dataset == "cifar10":
        _res_for_basis = 32
    else:
        _res_for_basis = 64

    # ===========================================================
    # NEW: teacher clustered PCA (highest priority if enabled)
    # ===========================================================
    if (args is not None) and getattr(args, "use_teacher_basis", False) and dataset == "imagenet":
        assert getattr(args, "teacher_basis_path", None) is not None, \
            "--teacher_basis_path is required when --use_teacher_basis"
        V_by_label, label2basis = load_teacher_clustered_basis(
            args.teacher_basis_path, device=device, res=_res_for_basis
        )
        V_global = None
        print("Loaded TEACHER clustered PCA basis (dict) + label2basis mapping.")
    else:
        # ---------- original logic (unchanged) ----------
        if os.path.exists(V_by_label_path):
            V_by_label = torch.load(V_by_label_path, map_location=device)
            print("Loaded per-label/bank PCA basis.")

            if os.path.exists(label2basis_path):
                with open(label2basis_path, "rb") as f:
                    label2basis = pickle.load(f)
                print(f"Loaded label2basis mapping from {label2basis_path}")
            else:
                print("No label2basis.pkl found (assuming true per-label PCA: basis_id == class_id).")
        else:
            V_global = torch.load(V_global_path, map_location=device)
            print("Loaded global PCA basis.")

    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading checkpoint: {ckpt_path}")
    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # --- load g / h depending on model_tag ---
    g_fns = {k: v.to(device) for k, v in ckpt["g_fns"].items()}
    h_fns = {k: v.to(device) for k, v in ckpt["h_fns"].items()}
    ema = ckpt["ema"].to(device).eval()
    print("Detected label-wise g_fn and h_fn checkpoint")

    # --- wrap net with basis-aware wrapper (STATELESS) ---
    edmnet = ANI_absM_Precond_Wrapper_basis_v2(
        ema,
        V_global=V_global,
        V_by_label=V_by_label,
    ).to(device).eval()

    print(f"Model loaded for {dataset} ({model_tag})")
    return edmnet, g_fns, h_fns, V_by_label, V_global, label2basis


# ===============================================================
# Sampling function
# ===============================================================
def sample_images(dataset="afhqv2", model_tag="g-ani", ckpt_root="finetune_initial", outdir_root="samples_rebuttal",
                  T=6400.0, seeds=range(50000), batch=2048, steps_list=[20], args=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    edmnet, g_fn, h_fn, V_by_label, V_global, label2basis = load_model(dataset, model_tag, ckpt_root, device, args=args)

    # resolution
    if dataset == "cifar10":
        res = 32
    else:
        res = 64

    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    for K in steps_list:
        print(f"--- Sampling {dataset} ({model_tag}) with {K} steps ---")
        t_vec = torch.linspace(0.0, T, K, device=device)
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-{K}")
        os.makedirs(subdir, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(seeds), batch), unit="batch"):
            batch_seeds = list(seeds[i:i + batch])

            latents = torch.stack([
                torch.randn([edmnet.img_channels, res, res],
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(int(s)))
                for s in batch_seeds
            ], dim=0)

            # ---------- class labels ----------
            if dataset in ["cifar10", "imagenet"]:
                rnd = torch.Generator(device=device).manual_seed(int(batch_seeds[0]))
                class_ids = torch.randint(
                    low=0, high=edmnet.label_dim,
                    size=(latents.shape[0],),
                    device=device, generator=rnd, dtype=torch.long
                )
                class_onehot = torch.nn.functional.one_hot(class_ids, num_classes=edmnet.label_dim).to(torch.float32)
            else:
                class_ids = None
                class_onehot = None

            # ---------- basis ids + V container ----------
            if V_by_label is not None:
                # basis bank case (ImageNet 1000 -> K clusters OR 32 banks)
                if label2basis is not None:
                    basis_ids = torch.tensor(
                        [int(label2basis[int(y.item())]) for y in class_ids],
                        device=device, dtype=torch.long
                    )
                else:
                    # true per-label PCA
                    basis_ids = class_ids

                V_container = V_by_label  # dict[basis_id -> V]
            else:
                basis_ids = None
                V_container = V_global     # tensor

            # ---------- generate (NO grouping, NO set_V_from_label) ----------
            with torch.no_grad():
                if isinstance(g_fn, dict):
                    g_batch = make_batch_fn(g_fn, class_ids, device)
                    h_batch = make_batch_fn(h_fn, class_ids, device)
                    imgs = M_EDM_ode_v3(
                        edmnet,
                        latents,
                        t_vec,
                        g_batch,
                        h_batch,
                        class_onehot,
                        V=V_container,
                        labels_id=basis_ids,
                    )
                else:
                    imgs = M_EDM_ode_v3(
                        edmnet,
                        latents,
                        t_vec,
                        g_fn,
                        h_fn,
                        class_onehot,
                        V=V_container,
                        labels_id=basis_ids,
                    )

            # save
            images_np = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()

            for seed, img_np in zip(batch_seeds, images_np):
                path = os.path.join(subdir, f"{int(seed):06d}.png")
                if img_np.shape[2] == 1:
                    PIL.Image.fromarray(img_np[:, :, 0], "L").save(path)
                else:
                    PIL.Image.fromarray(img_np, "RGB").save(path)


# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet", choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--model_tag", default="g-ani-rampup-10000-ema-50000-basis-v3-newpca-condition-ddp")
    parser.add_argument("--ckpt_root", default="finetune_initial")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=1500)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[20])
    parser.add_argument("--T", type=float, default=6400.0)

    # NEW (optional): teacher clustered PCA
    parser.add_argument("--use_teacher_basis", action="store_true",
                        help="Use teacher clustered PCA basis (Q + class_to_cluster) instead of basis_by_label.pt.")
    parser.add_argument("--teacher_basis_path", type=str, default="/hpc/group/chenglab/xc242/anisotropic-diffusion-icml/clustered_pca_via_classPCA128_K60_r1024_bestseed1.pt",
                        help="Path to teacher clustered PCA .pt file.")

    args = parser.parse_args()

    seed_range = range(args.start_seed, args.start_seed + args.num_seeds)
    sample_images(
        dataset=args.dataset,
        model_tag=args.model_tag,
        ckpt_root=args.ckpt_root,
        outdir_root=args.out,
        T=args.T,
        seeds=seed_range,
        batch=args.batch,
        steps_list=args.steps,
        args=args,   # NEW
    )
