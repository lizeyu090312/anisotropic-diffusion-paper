import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt, copy
from common_utils import GFn, compute_DCT_basis, M_EDM_ode, ANI_absM_Precond_Wrapper

torch.set_default_dtype(torch.float32)

# ===============================================================
# Load model (supports single g_fn or label-wise g_fns)
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device):
    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading checkpoint: {ckpt_path}")

    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # --------------------------------------------------
    # dataset meta
    # --------------------------------------------------
    if dataset == "cifar10":
        V_dim, res, num_classes = 256, 32, 10
        base_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
    elif dataset == "afhqv2":
        V_dim, res, num_classes = 1024, 64, 1
        base_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl"
    elif dataset == "ffhq":
        V_dim, res, num_classes = 1024, 64, 1
        base_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl"
    elif dataset == "imagenet":
        V_dim, res, num_classes = 1024, 64, 1000
        base_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"
    else:
        raise ValueError(dataset)

    # ==================================================
    # CASE 1: resume checkpoint (state_dict)
    # ==================================================
    if "resume" in model_tag:
        if "ani-condition" in model_tag:
            print("[load_model] RESUME checkpoint detected")
            normal_tag = 'g-ani-condition-v2-ddp-ema-50000-20k'
            normal_ckpt_path = os.path.join(
                ckpt_root, dataset, f"finetuned-{normal_tag}.pkl"
            )

            # ---- rebuild base EDM ----
            with dnnlib.util.open_url(normal_ckpt_path) as f:
                base = pickle.load(f)["ema"].to(device).eval()

            dct_V = compute_DCT_basis(
                k=V_dim, d=res, dtype=torch.float32, device=device
            )

            base.load_state_dict(ckpt["ema"], strict=True)
            edmnet = ANI_absM_Precond_Wrapper(base, dct_V).to(device).eval()

            # # ---- load EMA weights ----
            # edmnet.load_state_dict(ckpt["ema"], strict=True)

            # ---- load label-wise g / h ----
            if not os.path.exists(normal_ckpt_path):
                raise FileNotFoundError(
                    f"[ERROR] Normal checkpoint not found:\n{normal_ckpt_path}\n"
                    "Needed as GFn structure template for resume sampling."
                )

            with open(normal_ckpt_path, "rb") as f:
                normal_ckpt = pickle.load(f)

            # take any GFn as structure template
            if "g_fns" in normal_ckpt:
                g_template = next(iter(normal_ckpt["g_fns"].values()))
            else:
                g_template = normal_ckpt["g"]

            # ---- rebuild label-wise g / h ----
            g_fns = {}
            h_fns = {}

            for k, sd in ckpt["g_fns"].items():
                # ---- g_fn ----
                g = copy.deepcopy(g_template)
                g.load_state_dict(sd)
                g = g.to(device).eval()

                # ---- h_fn ----
                sd_h = ckpt["h_fns"][k]
                h = copy.deepcopy(g_template)
                h.load_state_dict(sd_h)
                h = h.to(device).eval()

                g_fns[str(k)] = g
                h_fns[str(k)] = h

            print("Loaded RESUME checkpoint successfully")
            return edmnet, g_fns, h_fns, res
        elif "iso-condition" in model_tag:
            print("[load_model] RESUME checkpoint detected")
            normal_tag = 'g-iso-condition-debug-continue'
            normal_ckpt_path = os.path.join(
                ckpt_root, dataset, f"finetuned-{normal_tag}.pkl"
            )

            # ---- rebuild base EDM ----
            with dnnlib.util.open_url(normal_ckpt_path) as f:
                base = pickle.load(f)["ema"].to(device).eval()

            dct_V = compute_DCT_basis(
                k=V_dim, d=res, dtype=torch.float32, device=device
            )

            base.load_state_dict(ckpt["ema"], strict=True)
            edmnet = ANI_absM_Precond_Wrapper(base, dct_V).to(device).eval()

            # # ---- load EMA weights ----
            # edmnet.load_state_dict(ckpt["ema"], strict=True)

            # ---- load label-wise g / h ----
            if not os.path.exists(normal_ckpt_path):
                raise FileNotFoundError(
                    f"[ERROR] Normal checkpoint not found:\n{normal_ckpt_path}\n"
                    "Needed as GFn structure template for resume sampling."
                )

            with open(normal_ckpt_path, "rb") as f:
                normal_ckpt = pickle.load(f)

            # take any GFn as structure template
            if "g_fns" in normal_ckpt:
                g_template = next(iter(normal_ckpt["g_fns"].values()))
            else:
                g_template = normal_ckpt["g"]

            # ---- rebuild label-wise g / h ----
            g_fns = {}
            h_fns = {}

            for k, sd in ckpt["g_fns"].items():
                # ---- g_fn ----
                g = copy.deepcopy(g_template)
                g.load_state_dict(sd)
                g = g.to(device).eval()

                # # ---- h_fn ----
                # sd_h = ckpt["h_fns"][k]
                # h = copy.deepcopy(g_template)
                # h.load_state_dict(sd_h)
                # h = h.to(device).eval()

                g_fns[str(k)] = g
                h_fns[str(k)] = g

            print("Loaded RESUME checkpoint successfully")
            return edmnet, g_fns, h_fns, res

    # ==================================================
    # CASE 2: normal checkpoint (Module objects)
    # ==================================================
    print("[load_model] NORMAL checkpoint detected")

    ema = ckpt["ema"].to(device).eval()

    # ---- detect schedule type ----
    if "ani-condition" in model_tag:
        g_fns = {str(k): v.to(device).eval() for k, v in ckpt["g_fns"].items()}
        h_fns = {str(k): v.to(device).eval() for k, v in ckpt["h_fns"].items()}
        print("Detected label-wise g_fn and h_fn checkpoint")

    elif "iso-condition" in model_tag:
        g_fns = {str(k): v.to(device).eval() for k, v in ckpt["g_fns"].items()}
        h_fns = g_fns
        print("Detected label-wise g_fn checkpoint")

    else:
        g_fns = ckpt["g"].to(device).eval()
        h_fns = ckpt.get("h", g_fns).to(device).eval()
        print("Detected single-schedule checkpoint")

    # ---- wrap with DCT basis ----
    dct_V = compute_DCT_basis(
        k=V_dim, d=res, dtype=torch.float32, device=device
    )
    edmnet = ANI_absM_Precond_Wrapper(ema, dct_V).to(device).eval()

    return edmnet, g_fns, h_fns, res


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
# Sampling
# ===============================================================
def sample_images(
    dataset="cifar10",
    model_tag="g-iso-condition",
    ckpt_root="finetune",
    outdir_root="samples",
    T=6400.0,
    seeds=range(50000),
    batch=2048,
    steps_list=[20],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    edmnet, g_fns, h_fns, res = load_model(dataset, model_tag, ckpt_root, device)

    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output dir: {outdir}")

    for K in steps_list:
        print(f"\n--- Sampling with {K} ODE steps ---")
        t_vec = torch.linspace(0.0, T, K, device=device)
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-{K}")
        os.makedirs(subdir, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(seeds), batch), unit="batch"):
            batch_seeds = seeds[i : i + batch]

            # --------------------------------
            # Latents
            # --------------------------------
            latents = torch.stack([
                torch.randn(
                    [edmnet.img_channels, res, res],
                    device=device,
                    generator=torch.Generator(device).manual_seed(s),
                )
                for s in batch_seeds
            ])

            # --------------------------------
            # Labels (conditional only)
            # --------------------------------
            if dataset in ["cifar10", "imagenet"]:
                rnd = torch.Generator(device=device).manual_seed(batch_seeds[0])
                class_ids = torch.randint(
                    low=0,
                    high=edmnet.label_dim,
                    size=(latents.shape[0],),
                    device=device,
                    generator=rnd,
                )
                labels = torch.eye(edmnet.label_dim, device=device)[class_ids]
            else:
                labels = None
                class_ids = None

            # --------------------------------
            # Sampling
            # --------------------------------
            with torch.no_grad():
                # ---- label-wise schedule ----
                if isinstance(g_fns, dict):
                    g_batch = make_batch_fn(g_fns, class_ids, device)
                    h_batch = make_batch_fn(h_fns, class_ids, device)
                    imgs = M_EDM_ode(
                        edmnet,
                        latents,
                        t_vec,
                        g_batch,
                        h_batch,
                        labels,
                    )

                else:
                    # ---- single schedule ----
                    imgs = M_EDM_ode(
                        edmnet,
                        latents,
                        t_vec,
                        g_fns,
                        h_fns,
                        labels,
                    )

            # --------------------------------
            # Save images
            # --------------------------------
            imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

            for seed, img in zip(batch_seeds, imgs):
                path = os.path.join(subdir, f"{seed:06d}.png")
                if img.shape[2] == 1:
                    PIL.Image.fromarray(img[:, :, 0], "L").save(path)
                else:
                    PIL.Image.fromarray(img, "RGB").save(path)


# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet",
                        choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--model_tag", default="g-ani-condition-v2-ddp-ema-50000-20k-resume-kimg-002000")
    parser.add_argument("--ckpt_root", default="finetune_initial")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=1500)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[6])
    parser.add_argument("--T", type=float, default=6400.0)
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
    )
