import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, compute_DCT_basis, M_EDM_ode, ANI_absM_Precond_Wrapper

torch.set_default_dtype(torch.float32)

# ===============================================================
# Helper: load model checkpoint (g-ani / g-h / g-iso)
# ===============================================================
def load_model(dataset, model_tag, device):
    ckpt_path = f"finetune/{dataset}/finetuned-{model_tag}.pkl"
    print(f"Loading checkpoint: {ckpt_path}")

    # --- load checkpoint ---
    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # --- load g / h depending on model_tag ---
    if "ani" in model_tag or "gh" in model_tag or "h" in model_tag:
        # anisotropic or gh models → expect both g and h
        g_fn = ckpt["g"].to(device)
        h_fn = ckpt["h"].to(device)
        print(f"Detected anisotropic model → using g and h schedules")
        ckpt_net_path = f"finetune/{dataset}/finetuned-g-ani.pkl"
        with dnnlib.util.open_url(ckpt_path) as f:
            ckpt_net = pickle.load(f)
        ema = ckpt_net["ema"].to(device).eval()
    else:
        # isotropic or g-only models → use g only
        g_fn = ckpt["g"].to(device)
        h_fn = g_fn
        print(f"Detected isotropic model → using g only (h = g)")
        ckpt_net_path = f"finetune/{dataset}/finetuned-g-iso.pkl"
        with dnnlib.util.open_url(ckpt_path) as f:
            ckpt_net = pickle.load(f)
        ema = ckpt_net["ema"].to(device).eval()


    # --- DCT basis setup ---
    if dataset == "cifar10":
        num_V_dim, res = 256, 32
    elif dataset in ["afhqv2", "ffhq"]:
        num_V_dim, res = 1024, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dct_V = compute_DCT_basis(k=num_V_dim, d=res, dtype=torch.float32, device=device)
    edmnet = ANI_absM_Precond_Wrapper(ema, dct_V).to(device)
    edmnet.eval()

    print(f"Model loaded for {dataset} ({model_tag}) at resolution {res}")
    return edmnet, g_fn, h_fn, res


# ===============================================================
# Sampling function
# ===============================================================
def sample_images(dataset="afhqv2", model_tag="g-ani", outdir_root="samples",
                  T=6400.0, seeds=range(50000), batch=2048, steps_list=[20]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    edmnet, g_fn, h_fn, res = load_model(dataset, model_tag, device)

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

            # conditional labels (CIFAR10 only)
            if dataset == "cifar10":
                rnd = torch.Generator(device=device).manual_seed(seeds[i])
                indices = torch.randint(low=0, high=edmnet.label_dim, size=(latents.shape[0],),
                                        device=device, generator=rnd)
                labels = torch.eye(edmnet.label_dim, device=device)[indices]
            else:
                labels = None

            # generate
            with torch.no_grad():
                imgs = M_EDM_ode(edmnet, latents, t_vec, g_fn, h_fn, labels)

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
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq"])
    parser.add_argument("--model_tag", default="g-ani", help="which model to sample: g-iso / g-ani / g-iso-wrapper / g-ani-wrapper")
    parser.add_argument("--out", default="samples")
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
                  outdir_root=args.out,
                  T=args.T,
                  seeds=seed_range,
                  batch=args.batch,
                  steps_list=args.steps)
