import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, compute_DCT_basis, ANI_absM_Precond_Wrapper
from generate import edm_sampler
torch.set_default_dtype(torch.float32)

class ANI_to_EDM_SigmaAdapter(torch.nn.Module):
    """
    Make ANI_absM_Precond_Wrapper look like an EDMPrecond-style network:
        forward(x, sigma, class_labels) -> D(x)
    by mapping sigma -> (g,h) = (sigma^2, sigma^2) (isotropic).
    """
    def __init__(self, ani_net: torch.nn.Module):
        super().__init__()
        self.ani = ani_net

        # Expose attributes that edm_sampler may query.
        for name in [
            "img_resolution", "img_channels", "label_dim",
            "use_fp16", "sigma_min", "sigma_max", "sigma_data"
        ]:
            if hasattr(ani_net, name):
                setattr(self, name, getattr(ani_net, name))

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **kwargs):
        # edm_sampler sometimes passes sigma as float / 0-d tensor / [B] tensor
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32)

        # Make it broadcast to batch: [B]
        B = x.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        elif sigma.ndim == 1 and sigma.shape[0] == 1:
            sigma = sigma.expand(B)
        elif sigma.ndim == 1 and sigma.shape[0] != B:
            # If sampler gives per-sample sigmas, it should match B; otherwise broadcast
            sigma = sigma[:1].expand(B)

        g = sigma ** 2
        h = sigma ** 2
        return self.ani(x, (g, h), class_labels=class_labels, force_fp32=force_fp32, **kwargs)

    def round_sigma(self, sigma):
        # edm_sampler may call this; just delegate if exists, else identity
        if hasattr(self.ani, "round_sigma"):
            return self.ani.round_sigma(sigma)
        return torch.as_tensor(sigma)

def make_karras_schedule(num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cuda'):
    t = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
    sigmas = (sigma_max**(1/rho) + t*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
    return sigmas.flip(0)
# ===============================================================
# Helper: load g/h from finetuned ckpt, but EDM net from NVLabs URL
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device):

    # -----------------------------------------------------------
    # 1) Load finetuned checkpoint (contains g/h schedules)
    # -----------------------------------------------------------
    # if "ani" in model_tag:
    #     ckpt_path = os.path.join(ckpt_root, dataset, "finetuned-g-ani-wrapper.pkl")
    #     print(f"Loading finetuned anisotropic checkpoint (g/h): {ckpt_path}")

    #     with dnnlib.util.open_url(ckpt_path) as f:
    #         ckpt = pickle.load(f)
    #     g_wrapped = ckpt["g"].to(device)   # c_wrapper_g
    #     h_wrapped = ckpt["h"].to(device)   # c_wrapper_h

    #     # ---------- geometric mean case ----------
    #     if "mean" in model_tag:
    #         print("Detected anisotropic + mean → using geometric-mean schedule via initg")

    #         g_disc_path = os.path.join('finetune_paper', dataset, "finetuned-g-iso-discretize.pkl")
    #         print(f"Loading discretization schedule grid from: {g_disc_path}")
    #         with dnnlib.util.open_url(g_disc_path) as f:
    #             disc_ckpt = pickle.load(f)
    #         g_grid = disc_ckpt["g"].to(device)

    #         times = g_grid.times              # (K,)
    #         T = float(g_grid.T.item())
    #         g0 = g_grid.g0

    #         g_vals, _ = g_wrapped.g_and_grad(times)   # shape (K,)
    #         h_vals, _ = h_wrapped.g_and_grad(times)   # shape (K,)

    #         g_geo_vals = torch.sqrt(g_vals * h_vals)

    #         g_geo = GFn(
    #             times=times,
    #             T=T,
    #             initg=g_geo_vals,
    #             g0=g0,
    #             device=device
    #         ).to(device)

    #         g_fn = g_geo
    #         h_fn = g_geo

    #     else:
    #         print("Detected anisotropic (no mean) → using wrapped g and h directly")
    #         g_fn = g_wrapped
    #         h_fn = h_wrapped

    # elif "karras" in model_tag:
    #     T=6400.0
    #     print(f"Loading karras schedule with K={K}, T={T}")
    #     sigmas = make_karras_schedule(num_steps=K, device=device)
    #     g_init = sigmas**2
    #     g_init[0], g_init[-1] = 1e-9, T
    #     times = torch.linspace(0.0, T, K, device=device)
    #     g_fn = GFn(times, T=T, initg=g_init, g0=1e-9).to(device)

    #     # For isotropic case, h = g
    #     h_fn = g_fn
    #     print(f"Detected karras → using g only (h = g)")
        
    # else:
        # ckpt_path = os.path.join(ckpt_root, dataset, "finetuned-g-iso-wrapper.pkl")
        # print(f"Loading finetuned isotropic checkpoint (g): {ckpt_path}")

        # with dnnlib.util.open_url(ckpt_path) as f:
        #     ckpt = pickle.load(f)
        # g_raw = ckpt["g"].to(device)
    # g_fn = g_raw
    # h_fn = g_raw
    # print("Detected isotropic → using h = g")

    # -----------------------------------------------------------
    # 2) Select NVLabs EDM pretrained URL for edmnet
    # -----------------------------------------------------------
    if dataset == 'cifar10':
        V_dim, res = 256, 32
    elif dataset == 'afhqv2':
        V_dim, res = 1024, 64
    elif dataset == 'ffhq':
        V_dim, res = 1024, 64
    elif dataset == 'imagenet':
        V_dim, res = 1024, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # -----------------------------------------------------------
    # 3) Load NVLabs EDM network (EMA)
    # -----------------------------------------------------------
    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading finetuned checkpoint (g): {ckpt_path}")

    with dnnlib.util.open_url(ckpt_path) as f:
        edm_ckpt = pickle.load(f)

    if "resume" in model_tag:
        print("[load_model] Detected RESUME checkpoint → rebuild model + load_state_dict")

        normal_tag = 'g-iso-condition-debug-continue'
        normal_ckpt_path = os.path.join(
            ckpt_root, dataset, f"finetuned-{normal_tag}.pkl"
        )

        # ---- rebuild base EDM ----
        with dnnlib.util.open_url(normal_ckpt_path) as f:
            edmnet = pickle.load(f)["ema"].to(device).eval()

        # # --- build ANI wrapper ---
        # dct_V = compute_DCT_basis(k=V_dim, d=res, device=device)
        # flow_net = ANI_absM_Precond_Flow_Net(base, dct_V).to(device).eval()

        # --- load EMA weights ---
        edmnet.load_state_dict(edm_ckpt["ema"], strict=True)

    # with dnnlib.util.open_url(ckpt_path) as f:
    #     edm_ckpt = pickle.load(f)
    else:
        edmnet = edm_ckpt["ema"].to(device).eval()

    # if isinstance(edmnet, ANI_absM_Precond_Wrapper):
    #     edmnet = ANI_to_EDM_SigmaAdapter(edmnet).to(device).eval()

    if hasattr(edmnet, "V") and hasattr(edmnet, "model"):
        print("Wrapping ANI edmnet with sigma adapter")
        edmnet = ANI_to_EDM_SigmaAdapter(edmnet).to(device).eval()

    # -----------------------------------------------------------
    # 4) Wrap with anisotropic preconditioner
    # -----------------------------------------------------------
    # dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)
    # edmnet = ANI_absM_Precond_Wrapper(edmnet, dct_V).to(device).eval()

    print(f"Loaded edmnet for {dataset}")
    return edmnet, res

# ===============================================================
# Sampling function
# ===============================================================
def sample_images(dataset="afhqv2", model_tag="g-ani", ckpt_root="finetune", outdir_root="samples",
                  T=6400.0, seeds=range(50000), batch=2048, steps_list=[20]):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # setup output dir
    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}-trained-edmsampler-seed-{args.start_seed}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")
    # load model
    edmnet, res = load_model(dataset, model_tag, ckpt_root, device)
    for K in steps_list:
        print(f"--- Sampling {dataset} ({model_tag}) with {K} steps ---")
        t_vec = torch.linspace(0.0, T, K, device=device)
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-trained-edmsampler-{K}")
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
            if dataset == "cifar10" or dataset == "imagenet":
                rnd = torch.Generator(device=device).manual_seed(seeds[i])
                indices = torch.randint(low=0, high=edmnet.label_dim, size=(latents.shape[0],),
                                        device=device, generator=rnd)
                labels = torch.eye(edmnet.label_dim, device=device)[indices]
            else:
                labels = None

            # generate
            with torch.no_grad():
                imgs = imgs = edm_sampler(edmnet,latents,num_steps=K,class_labels=labels)

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
    parser.add_argument("--model_tag", default="g-iso", help="which model to sample: g-iso / g-ani / g-iso-wrapper / g-ani-wrapper")
    parser.add_argument("--ckpt_root", default="finetune_initial", help="root folder containing finetuned checkpoints")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[18],
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