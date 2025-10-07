import argparse, pathlib, pickle, torch, tqdm, math, dnnlib, sys, copy
from torch_utils import misc, persistence
from common_utils import ANI_absM_Precond_Wrapper, ANILoss_gh_energy, GFn, compute_DCT_basis
from data_loader import cifar10_loader, afhqv2_loader, ffhq_loader

# ---------------- Utility ----------------
class DualWriter:
    """Print to both stdout and stderr (for HPC logging)."""
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

sys.stdout = DualWriter(sys.stdout, sys.stderr)

# ---------------- Schedules ----------------
def make_karras_schedule(num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cuda'):
    t = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
    sigmas = (sigma_max**(1/rho) + t*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
    return sigmas.flip(0)

# ---------------- Dataset factory ----------------
def get_dataset_loader(dataset, batch_size, workers=2):
    """
    Returns:
      loader: yields (imgs, labels_or_None)
      V_dim: number of DCT basis vectors
      res:   image resolution (32 or 64)
      ckpt_url: EDM pretrained URL (cond for CIFAR-10, uncond for others)
    """
    if dataset == 'cifar10':
        loader = cifar10_loader(batch_size, workers=workers)         # returns labels (int64)
        V_dim, res, ckpt_url = 256, 32, 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl'
    elif dataset == 'afhqv2':
        loader = afhqv2_loader(batch_size, workers=workers)          # labels=None
        V_dim, res, ckpt_url = 1024, 64, 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl'
    elif dataset == 'ffhq':
        loader = ffhq_loader(batch_size, workers=workers, down_to32=False)  # labels=None
        V_dim, res, ckpt_url = 1024, 64, 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl'
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return loader, V_dim, res, ckpt_url


def build_model_and_sched(ckpt_url, V_dim, res, device, T=6400.0, K=32):
    """
    Load pretrained EDM checkpoint, wrap both EMA and raw models into
    anisotropic preconditioners, and initialize scalar schedule g(t) (and h=g).

    Works for both cond (CIFAR-10) and uncond (AFHQ/FFHQ).
    """

    # --------------- Step 1. Compute DCT basis -----------------
    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    # --------------- Step 2. Load pretrained EDM weights --------
    with dnnlib.util.open_url(ckpt_url) as f:
        ckpt = pickle.load(f)

    # --------------- Step 3. Wrap EMA model ---------------------
    ema = ckpt['ema'].to(device).eval()
    ema = ANI_absM_Precond_Wrapper(ema, dct_V).to(device)

    # --------------- Step 4. Wrap raw model ---------------------
    # Some official ckpts do not have 'model' (only 'ema'); in that case, clone EMA
    raw = ckpt.get('model', pickle.loads(pickle.dumps(ema)))
    raw = ANI_absM_Precond_Wrapper(raw, dct_V).to(device)
    raw.train()
    for p in raw.parameters():
        p.requires_grad_(True)

    # --------------- Step 5. Build isotropic schedule g(t) ------
    sigmas = make_karras_schedule(num_steps=K, device=device)
    g_init = sigmas**2
    g_init[0], g_init[-1] = 1e-9, T
    times = torch.linspace(0.0, T, K, device=device)
    g_fn = GFn(times, T=T, initg=g_init, g0=1e-9).to(device)

    # For isotropic case, h = g
    h_fn = g_fn

    return raw, ema, g_fn, h_fn, dct_V


# ---------------- Train ----------------
def train(opt):
    device = torch.device('cuda')

    # per-dataset output dir
    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {outdir}")

    # data + model
    dl, V_dim, res, ckpt_url = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    model, ema, g_fn, h_fn, _ = build_model_and_sched(ckpt_url, V_dim, res, device)

    # optimizers
    opt_net   = torch.optim.Adam(model.parameters(), lr=opt.lr,  betas=(0.9, 0.999), eps=1e-8)
    opt_sched = torch.optim.Adam(g_fn.parameters(),  lr=1e-2,    betas=(0.9, 0.999), eps=1e-8)

    # training hyperparams
    T = 6400.0
    total_nimg        = opt.kimg * 1000
    lr_rampup_kimg    = 10_000
    ema_halflife_kimg = 500
    ema_rampup_ratio  = 0.05

    seen = 0
    loss_accum = 0.0
    batches_done = 0
    pbar = tqdm.tqdm(total=total_nimg, unit='img', dynamic_ncols=True)

    while seen < total_nimg:
        imgs, lbls = next(dl)              
        imgs = imgs.to(device)
        # CIFAR-10 likely in [0,1] → map to [-1,1]; AFHQ/FFHQ already in [-1,1]
        if opt.dataset == 'cifar10':
            imgs = imgs * 2.0 - 1.0
            lbls = lbls.to(device) # CIFAR-10: labels Tensor; AFHQ/FFHQ: None

        opt_net.zero_grad(set_to_none=True)
        opt_sched.zero_grad(set_to_none=True)

        # split into micro-batches (keep labels aligned)
        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        if lbls is None:
            y_chunks = [None] * len(x_chunks)
        else:
            y_chunks = torch.split(lbls, chunk)

        # forward/backward
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            loss = ANILoss_gh_energy(model, x_chunk, y_chunk, g_fn, h_fn, T=T, tmin=1e-9)
            (loss / opt.grad_accum).backward()
            loss_accum += loss.item()

        # sanitize grads
        for p in list(model.parameters()) + list(g_fn.parameters()):
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        # learning-rate schedule (ramp-up + exponential halves)
        ramp  = min(seen / max(lr_rampup_kimg * 1000, 1e-8), 1.0)  # convert kimg to img
        decay = 1.0 if seen <= total_nimg // 2 else 0.5 ** ((seen - total_nimg // 2) / (total_nimg // 8))
        lr_scale = ramp * decay
        opt_net.param_groups[0]['lr']   = opt.lr * lr_scale
        opt_sched.param_groups[0]['lr'] = 1e-2 * lr_scale

        for g in opt_net.param_groups:
            g['lr'] = opt.lr * lr_scale
        for g in opt_sched.param_groups:
            g['lr'] = 1e-2 * lr_scale

        # optimizer step
        if batches_done > 0:
            opt_net.step()
            opt_sched.step()

        # EMA update (with warmup on halflife)
        ema_halflife_nimg = ema_halflife_kimg * 1000
        ema_halflife_nimg = min(ema_halflife_nimg, seen * ema_rampup_ratio)
        ema_beta = 0.5 ** (opt.batch / max(ema_halflife_nimg, 1e-8))

        with torch.no_grad():
            for p_e, p_m in zip(ema.parameters(), model.parameters()):
                p_e.copy_(p_m.detach().lerp(p_e, ema_beta))

        # bookkeeping
        n = imgs.size(0)
        seen += n
        batches_done += 1
        pbar.update(n)

        if batches_done % 10 == 0:
            print(f"[{opt.dataset}] Seen {seen}/{total_nimg} | Avg loss {loss_accum / 10:.3e}")
            loss_accum = 0.0

        # periodic checkpoint
        if batches_done % 100 == 0:
            ckpt_path = outdir / f"finetuned-g-iso-ckpt-{batches_done:05d}.pkl"
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu()}, f)
            print(f"Saved checkpoint: {ckpt_path}")
            model.to(device); ema.to(device); g_fn.to(device)

    # final save
    final_path = outdir / "finetuned-g-iso.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu()}, f)
    print(f"Training finished. Final checkpoint: {final_path}")

# ---------------- Entry ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'afhqv2', 'ffhq'])
    parser.add_argument('--out', default='finetune')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kimg', type=int, default=1200)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    train(args)


