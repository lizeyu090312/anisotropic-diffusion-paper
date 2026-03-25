import argparse, pathlib, pickle, torch, tqdm, math, dnnlib, sys, copy
from torch_utils import misc, persistence
from common_utils import ANI_absM_Precond_Wrapper, ANILoss_gh_energy, GFn, compute_DCT_basis
from data_loader import cifar10_loader, afhqv2_loader, ffhq_loader, imagenet_loader

# ---------------- Utility ----------------
class DualWriter:
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
    if dataset == 'cifar10':
        loader = cifar10_loader(batch_size, workers=workers)
        V_dim, res, ckpt_url, num_classes = 256, 32, \
            'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl', 10
    elif dataset == 'afhqv2':
        loader = afhqv2_loader(batch_size, workers=workers)
        V_dim, res, ckpt_url, num_classes = 1024, 64, \
            'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl', 1
    elif dataset == 'ffhq':
        loader = ffhq_loader(batch_size, workers=workers, down_to32=False)
        V_dim, res, ckpt_url, num_classes = 1024, 64, \
            'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl', 1
    elif dataset == 'imagenet':
        loader = imagenet_loader(batch_size, workers=workers)  # labels=None
        V_dim, res, ckpt_url, num_classes = 1024, 64, 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl', 1000
    else:
        raise ValueError(dataset)

    return loader, V_dim, res, ckpt_url, num_classes

# ---------------- Model + sched ----------------
def build_model_and_sched(ckpt_url, V_dim, res, device, num_classes, dataset, T=6400.0, K=32):
    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    with dnnlib.util.open_url(ckpt_url) as f:
        ckpt = pickle.load(f)

    ema = ckpt['ema'].to(device).eval()
    ema = ANI_absM_Precond_Wrapper(ema, dct_V).to(device)

    raw = ckpt.get('model', pickle.loads(pickle.dumps(ema)))
    raw = ANI_absM_Precond_Wrapper(raw, dct_V).to(device)
    raw.train()
    if dataset == "imagenet":
        ema.use_fp16=False
        raw.use_fp16=False

    for p in raw.parameters():
        p.requires_grad_(True)
    sigmas = make_karras_schedule(num_steps=K, device=device)
    g_init = sigmas**2
    g_init[0], g_init[-1] = 1e-9, T
    times = torch.linspace(0.0, T, K, device=device)

    # ---------- label-wise g_fn ----------
    g_fns = torch.nn.ModuleDict({
        str(c): GFn(times, T=T, initg=g_init.clone(), g0=1e-9).to(device)
        for c in range(num_classes)
    })

    h_fns = g_fns  # isotropic

    return raw, ema, g_fns, h_fns

# ---------------- Train ----------------
def train(opt):
    device = torch.device('cuda')

    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    dl, V_dim, res, ckpt_url, num_classes = \
        get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)

    model, ema, g_fns, h_fns = \
        build_model_and_sched(ckpt_url, V_dim, res, device, num_classes, opt.dataset)

    # ---------- optimizers ----------
    opt_net = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8
    )

    opt_sched = {
        c: torch.optim.Adam(
            g_fns[str(c)].parameters(),
            lr=opt.glr, betas=(0.9, 0.999), eps=1e-8
        )
        for c in range(num_classes)
    }

    T = 6400.0
    total_nimg        = opt.kimg * 1000
    lr_rampup_kimg    = 10_000
    if opt.dataset == 'imagenet':
        ema_halflife_kimg = 50_000
    else:
        ema_halflife_kimg = 500
    ema_rampup_ratio  = args.ema_rampup_ratio

    seen, batches_done = 0, 0
    loss_accum = 0.0
    pbar = tqdm.tqdm(total=total_nimg, unit='img', dynamic_ncols=True)

    while seen < total_nimg:
        if batches_done == 0:
            ref = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if batches_done % 1000 == 0:
            with torch.no_grad():
                diff = 0.0
                denom = 0.0
                for k, v in model.state_dict().items():
                    diff += (v - ref[k]).abs().sum().item()
                    denom += ref[k].abs().sum().item() + 1e-12
                print(f"[probe] relative L1 change = {diff/denom:.3e}")

        imgs, lbls = next(dl)
        imgs = imgs.to(device)

        if opt.dataset == 'cifar10':
            imgs = imgs * 2.0 - 1.0
            lbls = lbls.to(device)
        elif opt.dataset == 'imagenet':
            lbls = lbls.to(device)
        else:
            lbls = torch.zeros(imgs.size(0), dtype=torch.long, device=device)

        opt_net.zero_grad(set_to_none=True)
        for o in opt_sched.values():
            o.zero_grad(set_to_none=True)

        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        if lbls is None:
            y_chunks = [None] * len(x_chunks)
        else:
            y_chunks = torch.split(lbls, chunk)

        used_labels = set()

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # ------------------------------------------------------------
            # Make class_ids: shape [B] int64, regardless of y_chunk format
            # ------------------------------------------------------------
            if y_chunk is None:
                class_ids = torch.zeros(x_chunk.size(0), dtype=torch.long, device=x_chunk.device)
                y_for_loss = None
            else:
                if y_chunk.ndim == 2:
                    # one-hot [B, C]
                    class_ids = y_chunk.argmax(dim=1)
                    y_for_loss = y_chunk  # keep one-hot for ANILoss_gh_energy
                else:
                    # integer labels [B]
                    class_ids = y_chunk
                    y_for_loss = y_chunk  # keep int labels

            # ------------------------------------------------------------
            # Split by class id -> pick corresponding g_fn optimizer
            # ------------------------------------------------------------
            for lbl in class_ids.unique():
                lbl_int = int(lbl.item())
                mask = (class_ids == lbl_int)  # [B] boolean

                x_lbl = x_chunk[mask]
                y_lbl = None if y_for_loss is None else y_for_loss[mask]

                if x_lbl.numel() == 0:
                    continue

                used_labels.add(lbl_int)

                loss = ANILoss_gh_energy(
                    model,
                    x_lbl,
                    y_lbl,
                    g_fns[str(lbl_int)],
                    h_fns[str(lbl_int)],
                    T=T,
                    tmin=1e-9
                )
                (loss / opt.grad_accum).backward()
                loss_accum += loss.item()

        # sanitize grads
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        for g in g_fns.values():
            for p in g.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)
        # LR schedule
        ramp = min(seen / max(lr_rampup_kimg * 1000, 1e-8), 1.0)
        decay = 1.0 if seen <= total_nimg // 2 else 0.5 ** ((seen - total_nimg // 2) / (total_nimg // 8))
        lr_scale = ramp * decay
        for g in opt_net.param_groups:
            g['lr'] = opt.lr * lr_scale
        for o in opt_sched.values():
            for pg in o.param_groups:
                pg["lr"] = opt.glr * lr_scale

        if batches_done > 0:
            opt_net.step()
            for lbl in used_labels:
                opt_sched[lbl].step()

        # ---------- EMA ----------
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, seen * ema_rampup_ratio)
        ema_beta = 0.5 ** (opt.batch / max(ema_halflife_nimg, 1e-8))

        with torch.no_grad():
            for p_e, p_m in zip(ema.parameters(), model.parameters()):
                p_e.copy_(p_m.detach().lerp(p_e, ema_beta))

        n = imgs.size(0)
        seen += n
        batches_done += 1
        pbar.update(n)

        if batches_done % 10 == 0:
            print(f"[{opt.dataset}] Seen {seen}/{total_nimg} | Avg loss {loss_accum / 10:.3e}")
            loss_accum = 0.0

        if batches_done % 100 == 0:
            ckpt_path = outdir / f"finetuned-g-iso-condition-debug.pkl"
            with open(ckpt_path, 'wb') as f:
                pickle.dump(
                    {
                        'model': model.cpu(),
                        'ema': ema.cpu(),
                        'g_fns': {k: v.cpu() for k, v in g_fns.items()},
                        'num_classes': num_classes
                    },
                    f
                )
            print(f"Saved checkpoint: {ckpt_path}")
            model.to(device)
            ema.to(device)
            for v in g_fns.values():
                v.to(device)

    ckpt_path = outdir / f"finetuned-g-iso-condition-debug.pkl"
    with open(ckpt_path, 'wb') as f:
        pickle.dump(
            {
                'model': model.cpu(),
                'ema': ema.cpu(),
                'g_fns': {k: v.cpu() for k, v in g_fns.items()},
                'num_classes': num_classes
            },
            f
        )

    print(f"Training finished.Final checkpoint: {ckpt_path}")

# ---------------- Entry ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'afhqv2', 'ffhq', "imagenet"])
    parser.add_argument('--out', default='finetune_initial')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--glr', type=float, default=1e-2)
    parser.add_argument('--kimg', type=int, default=1200)
    parser.add_argument('--ema_rampup_ratio', type=float, default=None)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)

    args = parser.parse_args()
    train(args)
