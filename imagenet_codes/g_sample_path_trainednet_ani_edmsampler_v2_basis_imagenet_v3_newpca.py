import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, mat_mul_labelwise
torch.set_default_dtype(torch.float32)

# ===============================================================
# Invert g(t) = y by bisection
# ===============================================================
@torch.no_grad()
def invert_g_bisect(g_fn, y, max_iter=32):
    y = torch.as_tensor(y, device=g_fn.times.device, dtype=torch.float32)
    y_min = torch.tensor(g_fn.g0, device=y.device, dtype=y.dtype)
    y_max = g_fn.T.to(device=y.device, dtype=y.dtype).reshape(())
    y = y.clamp(min=y_min, max=y_max)

    t_lo = torch.zeros_like(y)
    t_hi = torch.full_like(y, float(g_fn.T.item()))

    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        g_mid, _ = g_fn.g_and_grad(t_mid)
        t_lo = torch.where(g_mid < y, t_mid, t_lo)
        t_hi = torch.where(g_mid >= y, t_mid, t_hi)

    return 0.5 * (t_lo + t_hi)

# ===============================================================
# EDM-style sampler (Euler + Heun) with label-aware V support
# ===============================================================
def edm_flow_sampler(
    flow_net,
    latents,
    V,
    labels_id=None,
    class_labels=None,
    num_steps=18,
    rho=7,
    g_fn=None,
    h_fn=None,
    f_fn=None,
):
    sigma_min = 0.002
    sigma_max = 80

    device = latents.device
    B = latents.shape[0]

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = invert_g_bisect(f_fn, sigma_steps**2, max_iter=32)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0].sqrt()

    if isinstance(V, dict):
        assert labels_id is not None
        assert labels_id.dtype == torch.long
        max_key = max(V.keys())
        assert labels_id.min() >= 0 and labels_id.max() <= max_key

    for i, (t_hat, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat = x_next

        g_hat = g_fn(t_hat)[0].expand(B)
        h_hat = h_fn(t_hat)[0].expand(B)
        g_next = g_fn(t_next)[0].expand(B)
        h_next = h_fn(t_next)[0].expand(B)

        # Euler
        flow_hat = flow_net(
            x_hat,
            (g_hat, h_hat),
            V,
            labels_id=labels_id,
            class_labels=class_labels,
        )

        x_next = x_hat + mat_mul_labelwise(
            (g_next.sqrt() - g_hat.sqrt(),
             h_next.sqrt() - h_hat.sqrt()),
            V,
            -flow_hat,
            labels=labels_id,
            power=1.0,
            identity_scaling=0.0,
        )

        # Heun
        if i < num_steps - 1:
            flow_prime = flow_net(
                x_next,
                (g_next, h_next),
                V,
                labels_id=labels_id,
                class_labels=class_labels,
            )

            x_next = x_hat + mat_mul_labelwise(
                (g_next.sqrt() - g_hat.sqrt(),
                 h_next.sqrt() - h_hat.sqrt()),
                V,
                -(0.5 * flow_hat + 0.5 * flow_prime),
                labels=labels_id,
                power=1.0,
                identity_scaling=0.0,
            )

    return x_next

# ===============================================================
# NEW: teacher clustered PCA utils
# ===============================================================
@torch.no_grad()
def convert_Q_to_V(Q_2d, res):
    d, r = Q_2d.shape
    assert d == res * res
    V = Q_2d.T.reshape(r, res, res).contiguous()
    V = V / (V.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)).view(r, 1, 1)
    return V

@torch.no_grad()
def load_teacher_clustered_basis(path, device, res):
    bases = torch.load(path, map_location="cpu")
    Q_all = bases["Q"]                       # [K,4096,1024]
    class_to_cluster = bases["class_to_cluster"]

    V_by_label = {}
    for k in range(Q_all.shape[0]):
        V_by_label[k] = convert_Q_to_V(Q_all[k], res).to(device)

    label2basis = {i: int(class_to_cluster[i]) for i in range(len(class_to_cluster))}
    print(f"[Sampler] Loaded teacher clustered PCA: {len(V_by_label)} clusters")
    return V_by_label, label2basis

# ===============================================================
# Load model + schedules + PCA basis artifacts
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device, args):
    if dataset == 'cifar10':
        res = 32
    elif dataset in ['afhqv2', 'ffhq', 'imagenet']:
        res = 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading finetuned checkpoint: {ckpt_path}")
    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    flow_net = ckpt["ema"].to(device).eval()
    # g_fn = ckpt["g"].to(device).eval()
    # h_fn = ckpt["h"].to(device).eval()

    if "g" in ckpt:
        g_fn = ckpt["g"].to(device).eval()
    elif "g_fn" in ckpt:
        g_fn = ckpt["g_fn"].to(device).eval()
    else:
        raise KeyError("ckpt missing key 'g_fn'")

    if "h" in ckpt:
        h_fn = ckpt["h"].to(device).eval()
    elif "h_fn" in ckpt:
        h_fn = ckpt["h_fn"].to(device).eval()
    else:
        h_fn = g_fn

    f_fn = GFn(
        g_fn.times,
        g_fn.T,
        initg=(g_fn(g_fn.times)[0] * h_fn(h_fn.times)[0]) ** 0.5,
        g0=g_fn.g0,
        device=device,
    ).to(device).eval()

    basis_dir = os.path.join(ckpt_root, dataset)
    basis_by_label_path = os.path.join(basis_dir, "basis_by_label.pt")
    label2basis_path    = os.path.join(basis_dir, "label2basis.pkl")
    basis_global_path   = os.path.join(basis_dir, "basis_global.pt")

    V_by_label, V_global, label2basis = None, None, None

    # ---------- NEW: teacher clustered PCA ----------
    if dataset == "imagenet" and args.use_teacher_basis:
        V_by_label, label2basis = load_teacher_clustered_basis(
            args.teacher_basis_path, device=device, res=res
        )

    elif os.path.exists(basis_by_label_path) and os.path.exists(label2basis_path):
        V_by_label_cpu = torch.load(basis_by_label_path, map_location="cpu")
        with open(label2basis_path, "rb") as f:
            label2basis = pickle.load(f)
        V_by_label = {int(k): v.to(device) for k, v in V_by_label_cpu.items()}
        print(f"Loaded label-aware PCA basis bank with {len(V_by_label)} banks.")

    else:
        V_global = torch.load(basis_global_path, map_location=device)
        print("Loaded global PCA basis.")

    return flow_net, res, g_fn, h_fn, f_fn, V_by_label, V_global, label2basis

# ===============================================================
# Sampling
# ===============================================================
def sample_images(dataset="imagenet", model_tag="g-ani", ckpt_root="finetune_initial",
                  outdir_root="samples_rebuttal", T=6400.0,
                  seeds=range(50000), batch=2048, steps_list=[20], start_seed=0, args=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}-trained-ani-edmsampler-v2-seed-{start_seed}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    flow_net, res, g_fn, h_fn, f_fn, V_by_label, V_global, label2basis = load_model(
        dataset, model_tag, ckpt_root, device, args
    )

    for K in steps_list:
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-trained-ani-edmsampler-v2-{K}")
        os.makedirs(subdir, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(seeds), batch)):
            batch_seeds = list(seeds[i:i + batch])

            latents = torch.stack([
                torch.randn([3, res, res], device=device,
                            generator=torch.Generator(device).manual_seed(s))
                for s in batch_seeds
            ])

            labels = None
            labels_id = None

            if dataset in ["cifar10", "imagenet"]:
                rnd = torch.Generator(device=device).manual_seed(batch_seeds[0])
                indices = torch.randint(0, flow_net.label_dim,
                                        (latents.shape[0],),
                                        generator=rnd,
                                        device=device,
                                        dtype=torch.long)
                labels = torch.nn.functional.one_hot(
                    indices, num_classes=flow_net.label_dim
                ).float()

                if V_by_label is not None:
                    labels_id = torch.tensor(
                        [label2basis[int(y.item())] for y in indices],
                        device=device,
                        dtype=torch.long
                    )
                    V = V_by_label
                else:
                    V = V_global
            else:
                V = V_global

            with torch.no_grad():
                imgs = edm_flow_sampler(
                    flow_net,
                    latents,
                    V=V,
                    labels_id=labels_id,
                    class_labels=labels,
                    num_steps=K,
                    rho=7,
                    g_fn=g_fn,
                    h_fn=h_fn,
                    f_fn=f_fn,
                )

            imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

            for seed, img in zip(batch_seeds, imgs):
                PIL.Image.fromarray(img).save(os.path.join(subdir, f"{seed:06d}.png"))

# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--model_tag", default="g-ani-rampup-10000-ema-basis-v3-newpca")
    parser.add_argument("--ckpt_root", default="finetune_initial")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=1500)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[20])
    parser.add_argument("--T", type=float, default=6400.0)

    # NEW but optional
    parser.add_argument("--use_teacher_basis", action="store_true")
    parser.add_argument("--teacher_basis_path", type=str, default="/hpc/group/chenglab/xc242/anisotropic-diffusion-icml/clustered_pca_via_classPCA128_K60_r1024_bestseed1.pt")

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
        start_seed=args.start_seed,
        args=args,
    )
