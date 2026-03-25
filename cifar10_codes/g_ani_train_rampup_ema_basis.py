import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, copy, sys
from torch_utils import misc, persistence
from common_utils import GFn, flow_matching_energy_debug_basis, ANI_absM_Precond_Flow_Net_basis
from data_loader import afhqv2_loader, ffhq_loader, cifar10_loader, imagenet_loader
import os

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
        num_classes = 10
    elif dataset == "afhqv2":
        loader = afhqv2_loader(batch_size, workers=workers)
        V_dim, res = 1024, 64
        num_classes = 1
    elif dataset == "ffhq":
        loader = ffhq_loader(batch_size, workers=workers)
        V_dim, res = 1024, 64
        num_classes =1
    elif dataset == "imagenet":
        loader = imagenet_loader(batch_size, workers=workers)
        V_dim, res = 1024, 64
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return loader, V_dim, res, num_classes

# ---------------- PCA basis utils ----------------
@torch.no_grad()
def compute_pca_basis_from_loader(
    dataloader,
    k,
    d,
    device,
    max_batches=200,
    label=None,
    dataset="cifar10",
):
    X_list = []
    n_seen = 0

    for bi in range(max_batches):
        imgs, lbls = next(dataloader)

        if lbls.ndim > 1:
            lbls = lbls.argmax(dim=1)

        if label is not None:
            mask = (lbls == label)
            if mask.sum() == 0:
                continue
            imgs = imgs[mask]

        imgs = imgs.to(device=device, dtype=torch.float32)

        if dataset == "cifar10":
            imgs = imgs * 2 - 1

        # [B,C,H,W] -> [B,H,W]
        imgs = imgs.mean(dim=1)

        # [B,H,W] -> [B, H*W]
        X = imgs.reshape(imgs.shape[0], -1)
        X_list.append(X)
        n_seen += X.shape[0]

    X = torch.cat(X_list, dim=0)  # [N, d*d]
    X = X - X.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh[:k]  # [k, d*d]

    # reshape to [k, d, d]
    V = V.reshape(k, d, d).contiguous()

    V = V / (V.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)).reshape(k, 1, 1)

    return V

@torch.no_grad()
def build_V_by_label_pca(dl, *, num_classes, k, d, device, max_batches, dataset):
    V_by_label = {}
    for y in range(num_classes):
        print(f"[PCA] computing V for label={y}")
        V_by_label[y] = compute_pca_basis_from_loader(
            dl, k=k, d=d, device=device, max_batches=max_batches, label=y, dataset=dataset
        )
    return V_by_label

# ---------------- Model builder ----------------
def build_model_from_gonly(dataset, V_dim, res, device, out, dl_for_pca=None, num_classes=None, T=6400.0, K=32,
                          pca_batches=200, per_label_pca=False):
    """Load pretrained g-iso weights and initialize g-ani"""
    # gonly_path = os.path.join(out, dataset, f"finetuned-g-iso-rampup-10000-ema.pkl")
    if dataset == "imagenet":
        gonly_path = os.path.join(out, dataset, f"finetuned-g-iso-32-rampup-10000.pkl")
    else:
        gonly_path = os.path.join(out, dataset, f"finetuned-g-iso-rampup-10000-ema.pkl")
    print(f"Loading g-only initialization from {gonly_path}")
    with dnnlib.util.open_url(gonly_path, "rb") as f:
        ckpt = pickle.load(f)

    # --------- NEW: Build V via PCA ----------
    V_by_label = None
    if per_label_pca and (num_classes is not None) and (dl_for_pca is not None):
        V_by_label = build_V_by_label_pca(
            dl_for_pca, num_classes=num_classes, k=V_dim, d=res, device=device,
            max_batches=pca_batches, dataset=dataset
        )
        V0 = V_by_label[0]
    else:
        assert dl_for_pca is not None, "Need dl_for_pca to compute PCA basis"
        V0 = compute_pca_basis_from_loader(
            dl_for_pca, k=V_dim, d=res, device=device, max_batches=pca_batches, label=None, dataset=dataset
        )

    # --------- base nets ----------
    ema_base = ckpt["ema"].to(device)
    raw_base = ckpt["model"].to(device)
    g_fn = ckpt["g"].to(device)

    ema = ANI_absM_Precond_Flow_Net_basis(ema_base).to(device)
    raw = ANI_absM_Precond_Flow_Net_basis(raw_base).to(device)
    ema.eval(); raw.train()

    if dataset == 'imagenet':
        raw.use_fp16 = False
    for p in raw.parameters():
        p.requires_grad_(True)

    # initialize h = g (start from isotropic)
    times = torch.linspace(0.0, T, K, device=device)
    h_fn = GFn(times, T=T, initg=None).to(device)
    with open(gonly_path, "rb") as f:
        ckpt2 = pickle.load(f)
    h_fn = ckpt2['g'].to(device)

    print("Model built: g_ani1,g_ani2 initialized (from g-iso)")
    return raw, ema, g_fn, h_fn, V0, V_by_label

# ---------------- Train ----------------
def train(opt):
    device = torch.device("cuda")

    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {outdir}")
    print("Learnging rate:", opt.lr, opt.glr)

    # data
    dl, V_dim, res, num_classes = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    dl_pca, _, _, _ = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)

    # model
    model, ema, g_fn, h_fn, V0, V_by_label = build_model_from_gonly(
        opt.dataset, V_dim, res, device, opt.out,
        dl_for_pca=dl_pca,
        num_classes=num_classes,
        pca_batches=opt.pca_batches,
        per_label_pca=opt.per_label_pca
    )

    # after build_model_from_gonly(...)
    if V_by_label is not None:
        torch.save(
            {k: v.cpu() for k, v in V_by_label.items()},
            outdir / "basis_by_label.pt"
        )
    else:
        torch.save(V0.cpu(), outdir / "basis_global.pt")

    print("Saved PCA basis.")
    
    opt_model = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_sched = torch.optim.Adam(list(g_fn.parameters()) + list(h_fn.parameters()),
                                 lr=opt.glr, betas=(0.9, 0.999), eps=1e-8)

    T = 6400.0
    total_nimg = opt.kimg * 1000
    lr_rampup_kimg = opt.lr_rampup_kimg
    ema_halflife_kimg = 500
    ema_rampup_ratio = opt.ema_rampup_ratio

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
        elif opt.dataset == 'imagenet':
            lbls = lbls.to(device)
        else:
            lbls = None

        opt_model.zero_grad(set_to_none=True)
        opt_sched.zero_grad(set_to_none=True)

        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        # --- IMPORTANT: if per-label PCA, pass V_by_label into loss ---
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            loss = flow_matching_energy_debug_basis(
                model, x_chunk, y_chunk, g_fn, h_fn, opt.dataset, T=T, tmin=1e-9,
                V_by_label=V_by_label
            )
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

        if batches_done > 0:
            opt_model.step()
            opt_sched.step()

        # EMA
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
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

        if batches_done % 100 == 0:
            if opt.keep_all_ckpt:
                ckpt_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-basis-ckpt-{batches_done:05d}.pkl"
            else:
                ckpt_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-basis.pkl"
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu(), "h": h_fn.cpu()}, f)
            print(f"Saved: {ckpt_path}")
            model.to(device); ema.to(device); g_fn.to(device); h_fn.to(device)


    final_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-basis.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump({'model': model.cpu(), 'ema': ema.cpu(), 'g': g_fn.cpu(), "h": h_fn.cpu()}, f)
    print(f"Training finished. Final checkpoint: {final_path}")

# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--out", default="finetune_initial")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--glr", type=float, default=1e-4)
    parser.add_argument("--kimg", type=int, default=1200)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument('--lr_rampup_kimg', type=int, default=10000)
    parser.add_argument("--ema_rampup_ratio", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument('--keep_all_ckpt', action='store_true')
    parser.add_argument("--per_label_pca", action="store_true",
                        help="If set, compute one PCA basis per label and use V_y in the loss.")
    parser.add_argument("--pca_batches", type=int, default=200,
                        help="How many batches to use for PCA estimation.")

    args = parser.parse_args()
    train(args)
