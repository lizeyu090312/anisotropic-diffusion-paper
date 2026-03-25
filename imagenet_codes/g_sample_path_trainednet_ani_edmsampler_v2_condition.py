import math, torch, torch.nn as nn, pickle, dnnlib, tqdm, os, PIL, argparse, matplotlib.pyplot as plt, copy
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

def _make_f_fn(g_fn, h_fn, device):
    f_fn = GFn(
        g_fn.times,
        g_fn.T,
        initg=(g_fn(g_fn.times)[0] * h_fn(h_fn.times)[0])**0.5,
        g0=g_fn.g0,
        device=device
    ).eval()
    f_fn = f_fn.to(device)
    return f_fn

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
# Helper: load g/h from finetuned ckpt, but EDM net from NVLabs URL
# ===============================================================
def load_model(dataset, model_tag, ckpt_root, device):
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

    ckpt_path = os.path.join(ckpt_root, dataset, f"finetuned-{model_tag}.pkl")
    print(f"Loading checkpoint: {ckpt_path}")

    with dnnlib.util.open_url(ckpt_path) as f:
        ckpt = pickle.load(f)

    # ==================================================
    # CASE 1: resume checkpoint (state_dict)
    # ==================================================
    if "resume" in model_tag:
        print("[load_model] Detected RESUME checkpoint → rebuild model + load_state_dict")

        normal_tag = 'g-ani-condition-v2-ddp-ema-50000-20k'
        normal_ckpt_path = os.path.join(
            ckpt_root, dataset, f"finetuned-{normal_tag}.pkl"
        )

        # ---- rebuild base EDM ----
        with dnnlib.util.open_url(normal_ckpt_path) as f:
            flow_net = pickle.load(f)["ema"].to(device).eval()

        # # --- build ANI wrapper ---
        # dct_V = compute_DCT_basis(k=V_dim, d=res, device=device)
        # flow_net = ANI_absM_Precond_Flow_Net(base, dct_V).to(device).eval()

        # --- load EMA weights ---
        flow_net.load_state_dict(ckpt["ema"], strict=True)

        # --- load g / h (label-wise) ---
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

        f_fns = {
            k: _make_f_fn(g_fns[k], h_fns[k], device)
            for k in g_fns.keys()
        }
        return flow_net, res, g_fns, h_fns, f_fns, num_classes

    # ==================================================
    # CASE 2: normal checkpoint (Module objects)
    # ==================================================
    print("[load_model] Detected NORMAL checkpoint → direct load")

    assert "ema" in ckpt, f"{ckpt_path} must contain key 'ema'"
    flow_net = ckpt["ema"].to(device).eval()

    # --- label-wise schedules ---
    if "g_fns" in ckpt:
        g_fns = {str(k): v.to(device).eval() for k, v in ckpt["g_fns"].items()}
        h_fns = ckpt.get("h_fns", g_fns)
        h_fns = {str(k): v.to(device).eval() for k, v in h_fns.items()}

        f_fns = {
            k: _make_f_fn(g_fns[k], h_fns[k], device)
            for k in g_fns.keys()
        }

        return flow_net, res, g_fns, h_fns, f_fns, num_classes

    # --- single g / h ---
    assert "g" in ckpt, f"{ckpt_path} must contain key 'g'"
    g_fn = ckpt["g"].to(device).eval()
    h_fn = ckpt["h"].to(device).eval()
    f_fn = _make_f_fn(g_fn, h_fn, device)

    return flow_net, res, g_fn, h_fn, f_fn, num_classes

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
    flow_net, res, g_obj, h_obj, f_obj, num_classes = load_model(dataset, model_tag, ckpt_root, device)
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
                # imgs = edm_flow_sampler(flow_net, latents, class_labels=labels,num_steps=K, rho=7, 
                #                         g_fn=g_fn, h_fn=h_fn, f_fn=f_fn)
                # =====================================================
                # label-wise schedules: dict => split by class_ids
                # =====================================================
                if isinstance(g_obj, dict):
                    g_batch = make_batch_fn(g_obj, indices, device)
                    h_batch = make_batch_fn(h_obj, indices, device)
                    f_batch = make_batch_fn(f_obj, indices, device)

                    imgs = edm_flow_sampler(
                        flow_net,
                        latents,
                        class_labels=labels,
                        num_steps=K,
                        rho=7,
                        g_fn=g_batch,
                        h_fn=h_batch,
                        f_fn=f_batch,
                    )



                else:
                    # single schedule
                    imgs = edm_flow_sampler(
                        flow_net,
                        latents,
                        class_labels=labels,
                        num_steps=K,
                        rho=7,
                        g_fn=g_obj,
                        h_fn=h_obj,
                        f_fn=f_obj,
                    )
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
    parser.add_argument("--dataset", default="imagenet", choices=["cifar10", "afhqv2", "ffhq","imagenet"])
    parser.add_argument("--model_tag", default="g-ani-condition-v2-ddp-ema-50000-20k-resume-kimg-002000", help="which model to sample: g-iso / g-ani / g-iso-wrapper / g-ani-wrapper")
    parser.add_argument("--ckpt_root", default="finetune_initial", help="root folder containing finetuned checkpoints")
    parser.add_argument("--out", default="samples_rebuttal")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=50000)
    parser.add_argument("--steps", type=int, nargs="+", default=[5],
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