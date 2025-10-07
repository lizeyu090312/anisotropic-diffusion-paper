import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, copy, sys
from torch_utils import misc, persistence
from common_utils import GFn, compute_DCT_basis, flow_matching_energy, ANI_absM_Precond_Flow_Net
from data_loader import afhqv2_loader, ffhq_loader, cifar10_loader

# ---------------- Utility ----------------
class DualWriter:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
    def flush(self):
        self.stream1.flush(); self.stream2.flush()

sys.stdout = DualWriter(sys.stdout, sys.stderr)


# ---------------- Dataset factory ----------------
def get_dataset_loader(dataset, batch_size, workers=2):
    if dataset == "cifar10":
        loader = cifar10_loader(batch_size, workers=workers)
        V_dim, res = 256, 32
    elif dataset == "afhqv2":
        loader = afhqv2_loader(batch_size, workers=workers)
        V_dim, res = 1024, 64
    elif dataset == "ffhq":
        loader = ffhq_loader(batch_size, workers=workers)
        V_dim, res = 1024, 64
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return loader, V_dim, res


# ---------------- Model builder ----------------
def build_model_from_gonly(dataset, V_dim, res, device, T=6400.0, K=32):
    """Load pretrained g-iso weights and initialize g-ani"""
    gonly_path = f"finetune/{dataset}/finetuned-g-iso.pkl"
    print(f"Loading g-only initialization from {gonly_path}")
    with dnnlib.util.open_url(gonly_path, "rb") as f:
        ckpt = pickle.load(f)

    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    ema_base = ckpt["ema"].to(device)
    raw_base = ckpt["model"].to(device)
    g_fn = ckpt["g"].to(device)

    ema = ANI_absM_Precond_Flow_Net(ema_base, dct_V).to(device)
    raw = ANI_absM_Precond_Flow_Net(raw_base, dct_V).to(device)
    ema.eval(); raw.train()
    for p in raw.parameters():
        p.requires_grad_(True)

    # initialize h = g (start from isotropic)
    times = torch.linspace(0.0, T, K, device=device)
    h_fn = GFn(times, T=T, initg=None).to(device)
    with open(gonly_path, "rb") as f:
        ckpt = pickle.load(f)
    h_fn = ckpt['g'].to(device)

    print("Model built: g_ani1,g_ani2 initialized (from g-iso)")
    return raw, ema, g_fn, h_fn, dct_V


# ---------------- Train ----------------
def train(opt):
    device = torch.device("cuda")

    # setup output dir (same as g-iso)
    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {outdir}")

    # prepare data + model
    dl, V_dim, res = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    model, ema, g_fn, h_fn, _ = build_model_from_gonly(opt.dataset, V_dim, res, device)

    # optimizer
    opt_model = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_sched = torch.optim.Adam(list(g_fn.parameters()) + list(h_fn.parameters()),
                                 lr=opt.glr, betas=(0.9, 0.999), eps=1e-8)

    # training hyperparams
    T = 6400.0
    total_nimg = opt.kimg * 1000
    lr_rampup_kimg = 10_000
    ema_halflife_kimg = 500
    ema_rampup_ratio = 0.05

    seen = 0
    loss_accum = 0.0
    batches_done = 0
    pbar = tqdm.tqdm(total=total_nimg, unit="img", dynamic_ncols=True)

    while seen < total_nimg:
        imgs, lbls = next(dl)
        imgs = imgs.to(device)
        if opt.dataset == "cifar10":
            imgs = imgs * 2 - 1
            lbls = lbls.to(device)
        else:
            lbls = None

        opt_model.zero_grad(set_to_none=True)
        opt_sched.zero_grad(set_to_none=True)

        # split for grad accumulation
        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            loss = flow_matching_energy(model, x_chunk, y_chunk, g_fn, h_fn, T=T, tmin=1e-9)
            (loss / x_chunk.shape[0]).backward()
            loss_accum += (loss / x_chunk.shape[0]).item()

        # sanitize grads
        for p in list(model.parameters()) + list(g_fn.parameters()) + list(h_fn.parameters()):
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        # LR schedule
        ramp = min(seen / max(lr_rampup_kimg * 1000, 1e-8), 1.0)
        decay = 1.0 if seen <= total_nimg // 2 else 0.5 ** ((seen - total_nimg // 2) / (total_nimg // 8))
        lr_scale = ramp * decay
        for g in opt_model.param_groups:
            g['lr'] = opt.lr * lr_scale
        for g in opt_sched.param_groups:
            g['lr'] = opt.glr * lr_scale

        # optimizer step
        if batches_done > 0:
            opt_model.step()
            opt_sched.step()

        # EMA update
        ema_halflife_nimg = ema_halflife_kimg * 1000
        ema_halflife_nimg = min(ema_halflife_nimg, seen * ema_rampup_ratio)
        ema_beta = 0.5 ** (opt.batch / max(ema_halflife_nimg, 1e-8))
        with torch.no_grad():
            for p_e, p_m in zip(ema.parameters(), model.parameters()):
                p_e.copy_(p_m.detach().lerp(p_e, ema_beta))

        seen += imgs.size(0)
        batches_done += 1
        pbar.update(imgs.size(0))

        if batches_done % 10 == 0:
            avg = loss_accum / 10
            print(f"[{opt.dataset}] Seen {seen}/{total_nimg} | Avg loss {avg:.3e}")
            loss_accum = 0.0

        # checkpoint (same folder as g-only)
        if batches_done % 100 == 0:
            ckpt_path = outdir / f"finetuned-g-ani-ckpt-{batches_done:05d}.pkl"
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu(), "h": h_fn.cpu()}, f)
            print(f"Saved: {ckpt_path}")
            model.to(device); ema.to(device); g_fn.to(device); h_fn.to(device)

    # final save
    final_path = outdir / "finetuned-g-ani.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu(), "h": h_fn.cpu()}, f)
    print(f"Training finished. Final checkpoint: {final_path}")


# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq"])
    parser.add_argument("--out", default="finetune")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--glr", type=float, default=1e-4)  # g,h learning rate
    parser.add_argument("--kimg", type=int, default=1200)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train(args)
