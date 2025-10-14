import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, sys, os, random, string
from torch_utils import misc
from common_utils import (
    ANI_absM_Precond_Wrapper, ANILoss_gh_energy_plus_discretization,
    GFn, compute_DCT_basis, CNoiseProjectorBisect, c_wrapper_g, c_wrapper_h
)
from data_loader import cifar10_loader, afhqv2_loader, ffhq_loader


# ---------------- Logging ----------------
class DualWriter:
    def __init__(self, stream1, stream2, stream3):
        self.stream1, self.stream2, self.stream3 = stream1, stream2, stream3
    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
        self.stream3.write(message)
    def flush(self):
        self.stream1.flush(); self.stream2.flush(); self.stream3.flush()


# ---------------- Dataset selector ----------------
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
        raise ValueError(f"Unknown dataset: {dataset}")
    return loader, V_dim, res


# ---------------- Main training ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq"])
    parser.add_argument("--out", default="finetune")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--glr", type=float, default=1e-2)
    parser.add_argument("--kimg", type=int, default=2000)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument('--keep_all_ckpt', action='store_true', help='If set, keep all periodic checkpoints instead of overwriting the latest one.')

    opt = parser.parse_args()

    # ---------- Setup directories ----------
    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    log_file = outdir / "log_train_ani_wrapper.txt"
    log_stream = open(log_file, "w")
    sys.stdout = DualWriter(sys.stdout, sys.stderr, log_stream)
    print(f"=== Training started for dataset={opt.dataset}, folder={outdir} ===")

    device = torch.device("cuda")

    # ---------- Dataset + model ----------
    T = 6400.0
    dl, V_dim, res = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    # ---------- Load pretrained components ----------
    g_iso_path = f"finetune/{opt.dataset}/finetuned-g-ani.pkl"
    g_disc_path = f"finetune/{opt.dataset}/finetuned-g-iso-discretize.pkl"

    print(f"Loading g-ani checkpoint: {g_iso_path}")
    with dnnlib.util.open_url(g_iso_path) as f:
        ckpt = pickle.load(f)
        ema = ANI_absM_Precond_Wrapper(ckpt["ema"], dct_V).to(device)
        ema.eval()
        g_fn_o = ckpt["g"].to(device)
        h_fn_o = ckpt['h'].to(device)

    print(f"Loading g-iso schedule trained on discretization error: {g_disc_path}")
    with dnnlib.util.open_url(g_disc_path) as f:
        ckpt = pickle.load(f)
        g_wrapper = ckpt["g"].to(device)

    # ---------- Build projector and wrapped g,h ----------
    c_projector = CNoiseProjectorBisect(g_fn_o, h_fn_o, g_wrapper, bisect_iters=32)
    g_fn = c_wrapper_g(c_projector)
    h_fn = c_wrapper_h(c_projector)

    optimizer = torch.optim.Adam(list(c_projector.wrapper_parameters()), lr=opt.glr,
                                 betas=(0.9, 0.999), eps=1e-8)

    total_nimg = opt.kimg * 1000
    lr_rampup_kimg = 10
    seen = 0
    running_loss = 0.0
    batches_done = 0

    # ---------- Training loop ----------
    while seen < total_nimg:
        imgs, lbls = next(dl)
        imgs = imgs.to(device)

        if opt.dataset == 'cifar10':
            imgs = imgs * 2.0 - 1.0
            lbls = lbls.to(device)

        optimizer.zero_grad(set_to_none=True)

        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            loss = ANILoss_gh_energy_plus_discretization(ema, x_chunk, y_chunk, g_fn, h_fn, T, tmin=1e-9)
            (loss / x_chunk.shape[0]).backward()
            running_loss += (loss / x_chunk.shape[0]).item()

        for p in c_projector.wrapper_parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        # LR schedule
        ramp = min(seen / max(lr_rampup_kimg, 1e-8), 1.0)        # linear warm-up

        if seen <= total_nimg // 2:                             # no decay first 50 %
            decay = 1.0
        else:                                                    # halve every 12.5 %
            decay_steps = (seen - total_nimg // 2) / (total_nimg // 8)
            decay = 0.5 ** decay_steps                           # continuous 0.5^(Δ/0.125T)

        lr_scale = ramp * decay
        for g in optimizer.param_groups:
            g['lr'] = opt.glr * lr_scale
        optimizer.step()

        seen += imgs.size(0)
        batches_done += 1
        if batches_done % 10 == 0:
            avg = running_loss / 10
            print(f"[{opt.dataset}] img {seen}/{total_nimg} | loss {avg:.3e}")
            running_loss = 0.0

        # Save checkpoints periodically
        if batches_done % 50 == 0:
            if opt.keep_all_ckpt:
                ckpt_path = outdir / f"finetuned-g-ani-wrapper-ckpt-{batches_done:05d}.pkl"
            else:
                ckpt_path = outdir / "finetuned-g-ani-wrapper.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({"ema": ema.cpu(), "g": g_fn.cpu(), "h": h_fn.cpu()}, f)
            print("Saved", ckpt_path)
            ema.to(device); g_fn.to(device); h_fn.to(device)

    final_path = outdir / "finetuned-g-ani-wrapper.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"ema": ema.cpu(), "g": g_fn.cpu(), "h": h_fn.cpu()}, f)
    print(f"Training finished. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

