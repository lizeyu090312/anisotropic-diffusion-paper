import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt
from common_utils import GFn, compute_DCT_basis, ANI_absM_Precond_Flow_Net, mat_mul
torch.set_default_dtype(torch.float32)

def edm_flow_sampler(
    flow_net, latents, class_labels=None, num_steps=18, rho=7, g_fn=None, h_fn=None, f_fn=None
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = 0.002
    sigma_max = 80

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # convert sigma_s to time-index for f
    # note that f is by definition (g*h)**0.5
    t_steps = invert_g_bisect(f_fn, sigma_steps**2, max_iter=32)
    ## note to pengxi: might also be worth trying invert_g_bisect(g_fn...) and invert_g_bisect(h_fn...) instead of above line
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0].sqrt()
    dct_V = flow_net.V#.to(torch.float64)
    for i, (t_hat, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        g_hat = g_fn(t_hat)[0] * torch.ones_like(latents[:,0,0,0])
        h_hat = h_fn(t_hat)[0] * torch.ones_like(latents[:,0,0,0])
        g_next = g_fn(t_next)[0] * torch.ones_like(latents[:,0,0,0])
        h_next = h_fn(t_next)[0] * torch.ones_like(latents[:,0,0,0]) 
        # Euler step
        flow_hat = flow_net(x_hat, (g_hat, h_hat), class_labels)
        x_next = x_hat + mat_mul((g_next.sqrt() - g_hat.sqrt(), h_next.sqrt() - h_hat.sqrt()), 
                              dct_V, -flow_hat)
        if i < num_steps - 1:
            flow_prime = flow_net(x_next, (g_next, h_next), class_labels)
            x_next = x_hat + mat_mul((g_next.sqrt() - g_hat.sqrt(), h_next.sqrt() - h_hat.sqrt()), 
                              dct_V, -(0.5 * flow_hat + 0.5 * flow_prime))
    return x_next

@torch.no_grad()
def invert_g_bisect(g_fn, y, max_iter=32):
    """
    Solve g(t) = y for t in [0, T] by bisection.
    y: Tensor shape [B] (or broadcastable)
    returns t with same shape as y.
    """
    y = torch.as_tensor(y, device=g_fn.times.device, dtype=torch.float32)

    # clamp to valid range to avoid NaNs when sigma is out of range
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
# Helper: load g/h from finetuned ckpt, but EDM net from NVLabs URL
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device):
    # --- resolution / V_dim ---
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

    # --- load finetuned checkpoint that contains ema + g (and maybe h) ---
    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading finetuned checkpoint: {ckpt_path}")
    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # --- get ema net ---
    assert "ema" in ckpt, f"{ckpt_path} must contain key 'ema'"
    flow_net = ckpt["ema"].to(device).eval()

    # --- get g schedule ---
    assert "g" in ckpt, f"{ckpt_path} must contain key 'g'"
    g_fn = ckpt["g"].to(device).eval()
    h_fn = ckpt["h"].to(device).eval()
    f_fn = GFn(g_fn.times, g_fn.T, initg = (g_fn(g_fn.times)[0]*h_fn(h_fn.times)[0])**0.5, g0 = g_fn.g0, device=device)
    f_fn.to(device)

    print(f"Loaded model for {dataset}")
    return flow_net, res, g_fn, h_fn, f_fn


# ===============================================================
# Sampling function
# ===============================================================
def sample_images(dataset="afhqv2", model_tag="g-ani", ckpt_root="finetune", outdir_root="samples",
                  T=6400.0, seeds=range(50000), batch=2048, steps_list=[20]):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # setup output dir
    outdir = os.path.join(outdir_root, f"{dataset}-{model_tag}-trained-ani-edmsampler-v2-seed-{args.start_seed}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")
    # load model
    flow_net, res, g_fn, h_fn, f_fn = load_model(dataset, model_tag, ckpt_root, device)
    for K in steps_list:
        print(f"--- Sampling {dataset} ({model_tag}) with {K} steps ---")
        t_vec = torch.linspace(0.0, T, K, device=device)
        subdir = os.path.join(outdir, f"fid-heun-{model_tag}-trained-ani-edmsampler-v2-{K}")
        os.makedirs(subdir, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(seeds), batch), unit="batch"):
            batch_seeds = seeds[i:i + batch]
            latents = torch.stack([
                torch.randn([3, res, res],
                            device=device,
                            generator=torch.Generator(device).manual_seed(s))
                for s in batch_seeds
            ])

            # conditional labels (CIFAR10 only)
            if dataset == "cifar10" or dataset == "imagenet":
                rnd = torch.Generator(device=device).manual_seed(seeds[i])
                indices = torch.randint(low=0, high=flow_net.label_dim, size=(latents.shape[0],),
                                        device=device, generator=rnd)
                labels = torch.eye(flow_net.label_dim, device=device)[indices]
            else:
                labels = None

            # generate
            with torch.no_grad():
                imgs = edm_flow_sampler(flow_net, latents, class_labels=labels,num_steps=K, rho=7, 
                                        g_fn=g_fn, h_fn=h_fn, f_fn=f_fn)
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
    parser.add_argument("--model_tag", default="g-ani-rampup-10000", help="which model to sample: g-iso / g-ani / g-iso-wrapper / g-ani-wrapper")
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
