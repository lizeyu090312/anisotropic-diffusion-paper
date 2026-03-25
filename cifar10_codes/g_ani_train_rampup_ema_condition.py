import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, copy, sys
from torch_utils import misc, persistence
from common_utils import GFn, compute_DCT_basis, flow_matching_energy_debug, ANI_absM_Precond_Flow_Net
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
        V_dim, res, num_classes = 256, 32, 10
    elif dataset == "afhqv2":
        loader = afhqv2_loader(batch_size, workers=workers)
        V_dim, res, num_classes = 1024, 64, 1
    elif dataset == "ffhq":
        loader = ffhq_loader(batch_size, workers=workers)
        V_dim, res, num_classes = 1024, 64, 1
    elif dataset == "imagenet":
        loader = imagenet_loader(batch_size, workers=workers)
        V_dim, res, num_classes = 1024, 64, 1000
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return loader, V_dim, res, num_classes


# ---------------- Model builder ----------------
def build_model_from_gonly(dataset, V_dim, res, device, out, num_classes, T=6400.0, K=32):
    gonly_path = os.path.join(out, dataset, "finetuned-g-iso-condition-bug.pkl")
    print(f"Loading g initialization from {gonly_path}")

    with dnnlib.util.open_url(gonly_path, "rb") as f:
        ckpt = pickle.load(f)

    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    # -----------------------------
    # model
    # -----------------------------
    ema_base = ckpt["ema"].to(device)
    raw_base = ckpt["model"].to(device)

    ema = ANI_absM_Precond_Flow_Net(ema_base, dct_V).to(device)
    raw = ANI_absM_Precond_Flow_Net(raw_base, dct_V).to(device)
    ema.eval()
    raw.train()

    if dataset == "imagenet":
        raw.use_fp16 = False
        ema.use_fp16 = False

    for p in raw.parameters():
        p.requires_grad_(True)

    # -----------------------------
    # label-wise g / h initialization
    # -----------------------------
    g_fns = torch.nn.ModuleDict()
    h_fns = torch.nn.ModuleDict()

    if "g_fns" in ckpt:
        print("[Init] Using label-wise g_fns from g-iso-condition checkpoint")

        for c in range(num_classes):
            key = str(c)
            if key not in ckpt["g_fns"]:
                raise KeyError(f"Missing g_fns[{key}] in checkpoint")
            g_fns[key] = ckpt["g_fns"][key].to(device)
            h_fns[key] = copy.deepcopy(g_fns[key])  # start isotropic

    elif "g" in ckpt:
        print("[Init] Using single g and copying to all labels")

        g_base = ckpt["g"].to(device)
        for c in range(num_classes):
            g_fns[str(c)] = copy.deepcopy(g_base).to(device)
            h_fns[str(c)] = copy.deepcopy(g_base).to(device)

    else:
        raise KeyError("Checkpoint must contain either `g_fns` or `g`")

    print(f"Model built with label-wise g/h (num_classes={num_classes})")
    return raw, ema, g_fns, h_fns, dct_V


# ---------------- Train ----------------
def train(opt):
    device = torch.device("cuda")

    # setup output dir (same as g-iso)
    outdir = pathlib.Path(opt.out) / opt.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {outdir}")

    print("Learnging rate:", opt.lr, opt.glr)

    # prepare data + model
    dl, V_dim, res, num_classes = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    model, ema, g_fns, h_fns, _ = build_model_from_gonly(opt.dataset, V_dim, res, device, opt.out, num_classes)

    # optimizer
    opt_model = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_sched = {
        c: torch.optim.Adam(
            list(g_fns[str(c)].parameters()) + list(h_fns[str(c)].parameters()),
            lr=opt.glr, betas=(0.9, 0.999), eps=1e-8
        )
        for c in range(num_classes)
    }

    # training hyperparams
    T = 6400.0
    total_nimg = opt.kimg * 1000
    lr_rampup_kimg = 10000
    # ema_halflife_kimg = 500
    if opt.dataset == 'imagenet':
        ema_halflife_kimg = 50000
    else:
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
        for o in opt_sched.values():
            o.zero_grad(set_to_none=True)

        # split for grad accumulation
        chunk = max(1, opt.batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        used_labels = set()

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):

            # ------------------------------------
            # build class_ids [B]
            # ------------------------------------
            if y_chunk is None:
                class_ids = torch.zeros(x_chunk.size(0), dtype=torch.long, device=x_chunk.device)
                y_for_loss = None
            else:
                if y_chunk.ndim == 2:
                    # one-hot
                    class_ids = y_chunk.argmax(dim=1)
                    y_for_loss = y_chunk
                else:
                    class_ids = y_chunk
                    y_for_loss = y_chunk

            # ------------------------------------
            # split by label
            # ------------------------------------
            for lbl in class_ids.unique():
                lbl_int = int(lbl.item())
                mask = (class_ids == lbl_int)

                x_lbl = x_chunk[mask]
                y_lbl = None if y_for_loss is None else y_for_loss[mask]

                if x_lbl.numel() == 0:
                    continue

                used_labels.add(lbl_int)

                loss = flow_matching_energy_debug(
                    model,
                    x_lbl,
                    y_lbl,
                    g_fns[str(lbl_int)],
                    h_fns[str(lbl_int)],
                    opt.dataset,
                    T=T,
                    tmin=1e-9
                )
                (loss / opt.grad_accum).backward()
                loss_accum += (loss / x_chunk.shape[0]).item()

        # sanitize grads
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        for g in g_fns.values():
            for p in g.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        for h in h_fns.values():
            for p in h.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1e5, neginf=-1e5)

        # LR schedule
        ramp = min(seen / max(lr_rampup_kimg * 1000, 1e-8), 1.0)
        decay = 1.0 if seen <= total_nimg // 2 else 0.5 ** ((seen - total_nimg // 2) / (total_nimg // 8))
        lr_scale = ramp * decay
        for g in opt_model.param_groups:
            g['lr'] = opt.lr * lr_scale
        for o in opt_sched.values():
            for pg in o.param_groups:
                pg["lr"] = opt.glr * lr_scale

        # optimizer step
        if batches_done > 0:
            opt_model.step()
            for lbl in used_labels:
                opt_sched[lbl].step()

        # EMA update
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

        # checkpoint (same folder as g-only)
        if batches_done % 100 == 0:
            if opt.keep_all_ckpt:
                ckpt_path = outdir / f"finetuned-g-ani-condition-ckpt-{batches_done:05d}.pkl"
            else:
                ckpt_path = outdir / f"finetuned-g-ani-condition.pkl"
            with open(ckpt_path, 'wb') as f:
                pickle.dump(
                    {
                        'model': model.cpu(),
                        'ema': ema.cpu(),
                        'g_fns': {k: v.cpu() for k, v in g_fns.items()},
                        'h_fns': {k: v.cpu() for k, v in h_fns.items()},
                    },
                    f
                )

            print(f"Saved: {ckpt_path}")
            model.to(device); ema.to(device)
            for v in g_fns.values(): v.to(device)
            for v in h_fns.values(): v.to(device)


    # final save
    final_path = outdir / f"finetuned-g-ani-condition.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(
            {
                'model': model.cpu(),
                'ema': ema.cpu(),
                'g_fns': {k: v.cpu() for k, v in g_fns.items()},
                'h_fns': {k: v.cpu() for k, v in h_fns.items()},
            },
            f
        )

    print(f"Training finished. Final checkpoint: {final_path}")


# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--out", default="finetune_initial")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--glr", type=float, default=1e-4)  # g,h learning rate
    parser.add_argument("--kimg", type=int, default=1200)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--ema_rampup_ratio", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument('--keep_all_ckpt', action='store_true', help='If set, keep all periodic checkpoints instead of overwriting the latest one.')

    args = parser.parse_args()
    train(args)
