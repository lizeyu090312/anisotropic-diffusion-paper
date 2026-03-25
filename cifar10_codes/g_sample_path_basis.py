import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, GFnSelector, compute_DCT_basis, M_EDM_ode, ANI_absM_Precond_Wrapper_basis


torch.set_default_dtype(torch.float32)

# ===============================================================
# Helper: load model checkpoint (g-ani / g-h / g-iso)
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device):
    basis_dir = os.path.join(ckpt_root, dataset)

    V_by_label_path = os.path.join(basis_dir, "basis_by_label.pt")
    V_global_path   = os.path.join(basis_dir, "basis_global.pt")

    if os.path.exists(V_by_label_path):
        V_by_label = torch.load(V_by_label_path, map_location=device)
        V_global = None
        print("Loaded per-label PCA basis.")
    else:
        V_global = torch.load(V_global_path, map_location=device)
        V_by_label = None
        print("Loaded global PCA basis.")

    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading checkpoint: {ckpt_path}")

    # --- load checkpoint ---
    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # --- load g / h depending on model_tag ---
    if "ani-edm" in model_tag:
        # --- load g only ---
        g_fn = ckpt["g"].to(device)
        # --- derive h deterministically: h(t) = t^2 / g(t) ---
        class HFromG(nn.Module):
            def __init__(self, g_fn):
                super().__init__()
                self.g_fn = g_fn

            def g_and_grad(self, t):
                """
                h(t) = t^2 / g(t)
                dh/dt = 2t/g(t) - t^2/g(t)^2 * g'(t)
                """
                g, g_dot = self.g_fn.g_and_grad(t)
                h = (t**2) / g
                h_dot = (2 * t / g) - (t**2 / g**2) * g_dot
                return h, h_dot

            def forward(self, t):
                h, _ = self.g_and_grad(t)
                return h

        h_fn = HFromG(g_fn)
        print("Detected anisotropic EDM model → using g(t) and h(t)=t^2/g(t)")
        # --- load EMA network ---
        ckpt_net_path = os.path.join(ckpt_root, dataset, "finetuned-g-ani-edm.pkl")
        with dnnlib.util.open_url(ckpt_net_path) as f:
            ckpt_net = pickle.load(f)
        ema = ckpt_net["ema"].to(device).eval()

    elif "range" in model_tag:
        print("Detected anisotropic RANGE model → rebuilding g_fn_used")

        # --- load pieces ---
        g_full = ckpt["g_full"].to(device)
        g_mid  = ckpt["g_mid"].to(device)
        h_full = ckpt["h_full"].to(device)
        h_mid  = ckpt["h_mid"].to(device)

        t_lo = ckpt["t_lo"]
        t_hi = ckpt["t_hi"]

        # --- rebuild selector ---
        g_fn = GFnSelector(g_full, g_mid, t_lo, t_hi).to(device)
        h_fn = GFnSelector(h_full, h_mid, t_lo, t_hi).to(device)

        ema = ckpt["ema"].to(device).eval()

        print(f"Rebuilt g_fn_used with interval [{t_lo:.2f}, {t_hi:.2f}]")

    elif "ani-rampup" in model_tag:
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["h"].to(device)
        print(f"Detected anisotropic model → using g and h schedules and edmnet from {ckpt_path}")
        ema = ckpt["ema"].to(device).eval()

    elif "iso-rampup" in model_tag:
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["g"].to(device)
        print(f"Detected isotropic model → using g schedules and edmnet from {ckpt_path}")
        ema = ckpt["ema"].to(device).eval()

    elif "iso-32-rampup" in model_tag:
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["g"].to(device)
        print(f"Detected isotropic model → using g schedules and edmnet from {ckpt_path}")
        ema = ckpt["ema"].to(device).eval()

    elif "ani-32-rampup" in model_tag:
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["h"].to(device)
        print(f"Detected anisotropic model → using g and h schedules and edmnet from {ckpt_path}")
        ema = ckpt["ema"].to(device).eval()

    elif "ani" in model_tag or "gh" in model_tag:
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["h"].to(device)
        print(f"Detected anisotropic model → using g and h schedules")
        ckpt_net_path = os.path.join(ckpt_root, dataset, "finetuned-g-ani.pkl")
        with dnnlib.util.open_url(ckpt_net_path) as f:
            ckpt_net = pickle.load(f)
        ema = ckpt_net["ema"].to(device).eval()
    else:
        if "bp" in model_tag:
            print(f"Detected isotropic wrapper model (backprop) → using wrapper g only (h = g) and trained net")
            ckpt_net_path = ckpt_path
            with dnnlib.util.open_url(os.path.join(ckpt_root, dataset, f"finetuned-g-iso-wrapper.pkl")) as f:
                ckpt_wrapper = pickle.load(f)
            g_fn = ckpt_wrapper["g"].to(device)
            h_fn = g_fn
        elif "newg" in model_tag or "edmloss" in model_tag:
            print(f"Detected isotropic model → using iso g only (h = g) and trained net")
            ckpt_net_path = ckpt_path
            with dnnlib.util.open_url(os.path.join(ckpt_root, dataset, f"finetuned-g-iso.pkl")) as f:
                ckpt_wrapper = pickle.load(f)
            g_fn = ckpt_wrapper["g"].to(device)
            h_fn = g_fn
        elif "iso-wrapper-rampup-10000-ema" in model_tag:
            print(f"Detected isotropic wrapper model → using wrapper g only (h = g) from {ckpt_path} and net from finetuned-g-iso-rampup-10000-ema.pkl")
            ckpt_net_path = os.path.join(ckpt_root, dataset, "finetuned-g-iso-rampup-10000-ema.pkl")
            g_fn = ckpt["g"].to(device)
            h_fn = ckpt["g"].to(device)
        else:
            print(f"Detected isotropic model → using g only (h = g)")
            ckpt_net_path = os.path.join(ckpt_root, dataset, "finetuned-g-iso.pkl")
            g_fn = ckpt["g"].to(device)
            h_fn = g_fn
        with dnnlib.util.open_url(ckpt_net_path) as f:
            ckpt_net = pickle.load(f)
        ema = ckpt_net["ema"].to(device).eval()

    # --- DCT basis setup ---
    if dataset == "cifar10":
        num_V_dim, res = 256, 32
    elif dataset in ["afhqv2", "ffhq", "imagenet"]:
        num_V_dim, res = 1024, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # dct_V = compute_DCT_basis(k=num_V_dim, d=res, dtype=torch.float32, device=device)
    # edmnet = ANI_absM_Precond_Wrapper(ema, dct_V).to(device)
    edmnet = ANI_absM_Precond_Wrapper_basis(
                ema,
                V_global=V_global,
                V_by_label=V_by_label,
            ).to(device).eval()

    edmnet.eval()

    print(f"Model loaded for {dataset} ({model_tag}) at resolution {res}")
    return edmnet, g_fn, h_fn, res, V_by_label, V_global


# ===============================================================
# Sampling function
# ===============================================================
def sample_images(dataset="afhqv2", model_tag="g-ani", ckpt_root="finetune", outdir_root="samples",
                  T=6400.0, seeds=range(50000), batch=2048, steps_list=[20]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    edmnet, g_fn, h_fn, res, V_by_label, _ = load_model(dataset, model_tag, ckpt_root, device)

    # setup output dir
    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    for K in steps_list:
        print(f"--- Sampling {dataset} ({model_tag}) with {K} steps ---")
        t_vec = torch.linspace(0.0, T, K, device=device)
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-{K}")
        os.makedirs(subdir, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(seeds), batch), unit="batch"):
            batch_seeds = seeds[i:i + batch]

            latents = torch.stack([
                torch.randn([edmnet.img_channels, res, res],
                            device=device,
                            generator=torch.Generator(device).manual_seed(s))
                for s in batch_seeds
            ])

            # labels
            if dataset in ["cifar10", "imagenet"]:
                rnd = torch.Generator(device=device).manual_seed(seeds[i])
                indices = torch.randint(
                    low=0, high=edmnet.label_dim,
                    size=(latents.shape[0],),
                    device=device, generator=rnd, dtype=torch.long
                )
                labels = torch.eye(edmnet.label_dim, device=device)[indices]
            else:
                indices = None
                labels = None

            # generate
            with torch.no_grad():
                if (V_by_label is None) or (labels is None):
                    imgs = M_EDM_ode(edmnet, latents, t_vec, g_fn, h_fn, labels)
                else:
                    imgs = torch.empty_like(latents)
                    for y in torch.unique(indices):
                        y_int = int(y.item())
                        mask = (indices == y_int)
                        if mask.sum() == 0:
                            continue

                        edmnet.set_V_from_label(y_int)
                        imgs_sub = M_EDM_ode(edmnet, latents[mask], t_vec, g_fn, h_fn, labels[mask])
                        imgs[mask] = imgs_sub

            # save
            images_np = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()

            for seed, img_np in zip(batch_seeds, images_np):
                path = os.path.join(subdir, f"{seed:06d}.png")
                if img_np.shape[2] == 1:
                    PIL.Image.fromarray(img_np[:, :, 0], "L").save(path)
                else:
                    PIL.Image.fromarray(img_np, "RGB").save(path)


# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq","imagenet"])
    parser.add_argument("--model_tag", default="g-ani-rampup-10000-ema-basis", help="which model to sample: g-iso / g-ani / g-iso-wrapper / g-ani-wrapper")
    parser.add_argument("--ckpt_root", default="finetune_initial", help="root folder containing finetuned checkpoints")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[20],
                        help="ODE step counts (e.g. --steps 10 20 40 60)")
    parser.add_argument("--T", type=float, default=6400.0)
    args = parser.parse_args()

    seed_range = range(args.start_seed, args.start_seed + args.num_seeds)
    sample_images(dataset=args.dataset,
                  model_tag=args.model_tag,
                  ckpt_root=args.ckpt_root,
                  outdir_root=args.out,
                  T=args.T,
                  seeds=seed_range,
                  batch=args.batch,
                  steps_list=args.steps)
