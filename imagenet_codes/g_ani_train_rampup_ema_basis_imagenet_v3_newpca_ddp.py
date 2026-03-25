import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, copy, sys, os
from torch_utils import misc, persistence
from common_utils import GFn, flow_matching_energy_debug_basis_v3, ANI_absM_Precond_Flow_Net_basis_v3
from data_loader import afhqv2_loader, ffhq_loader, cifar10_loader, imagenet_loader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Utility ----------------
def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world():
    return dist.get_world_size() if ddp_is_initialized() else 1

def is_main():
    return get_rank() == 0

class DualWriter:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
    def flush(self):
        self.stream1.flush(); self.stream2.flush()

def setup_ddp():
    """Initialize torch.distributed and set correct GPU for this process."""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

def barrier(device_ids=None):
    if ddp_is_initialized():
        if device_ids is None:
            dist.barrier()
        else:
            dist.barrier(device_ids=device_ids)

def broadcast_module_params(module, src=0):
    """Broadcast all parameters/buffers of a module from src to all ranks."""
    if not ddp_is_initialized():
        return
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src=src)

def allreduce_grads(module):
    """
    All-reduce (sum) grads, then average by world size.
    IMPORTANT: never skip grad=None; create zeros so every rank participates.
    """
    if not ddp_is_initialized():
        return
    world = get_world()
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)

# sys.stdout = DualWriter(sys.stdout, sys.stderr)

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
        num_classes = 1
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
    for bi in range(max_batches):
        imgs, lbls = next(dataloader)

        if lbls.ndim > 1:
            lbls = lbls.argmax(dim=1)

        if label is not None:
            if isinstance(label, (list, tuple)):
                lab = torch.as_tensor(label, device=lbls.device, dtype=lbls.dtype)
                mask = torch.isin(lbls, lab)
            else:
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

    X = torch.cat(X_list, dim=0)  # [N, d*d]
    X = X - X.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh[:k]  # [k, d*d]

    # reshape to [k, d, d]
    V = V.reshape(k, d, d).contiguous()
    V = V / (V.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)).reshape(k, 1, 1)
    return V

@torch.no_grad()
def build_V_by_label_pca(dl, *, num_classes, k, d, device, max_batches, dataset, num_banks=32):
    """
    Label-aware basis bank:
      basis_id = y % num_banks
      V_by_label[basis_id] is PCA basis computed from a *set* of labels in that bank.
    Returns:
      V_by_label: dict[basis_id] -> V
      label2basis: dict[label] -> basis_id
    """
    V_by_label = {}
    label2basis = {y: (y % num_banks) for y in range(num_classes)}

    for b in range(num_banks):
        dl_iter = iter(dl)  # IMPORTANT: fresh iterator per bank
        labels_in_bank = [y for y in range(num_classes) if label2basis[y] == b]
        print(f"[PCA] computing V for basis_bank={b} (#labels={len(labels_in_bank)})")
        V_by_label[b] = compute_pca_basis_from_loader(
            dl_iter, k=k, d=d, device=device, max_batches=max_batches, label=labels_in_bank, dataset=dataset
        )
    return V_by_label, label2basis

def _try_load_saved_pca(outdir: pathlib.Path, per_label_pca: bool):
    """Try load PCA artifacts from disk. Return (V0, V_by_label, label2basis) or (None, None, None)."""
    basis_by_label = outdir / "basis_by_label.pt"
    label2basis_pkl = outdir / "label2basis.pkl"
    basis_global = outdir / "basis_global.pt"

    if per_label_pca:
        if basis_by_label.is_file() and label2basis_pkl.is_file():
            print(f"[PCA] Loading existing label-aware PCA from: {basis_by_label} and {label2basis_pkl}")
            V_by_label_cpu = torch.load(basis_by_label, map_location="cpu")
            with open(label2basis_pkl, "rb") as f:
                label2basis = pickle.load(f)
            V0 = next(iter(V_by_label_cpu.values()))
            return V0, V_by_label_cpu, label2basis
        return None, None, None
    else:
        if basis_global.is_file():
            print(f"[PCA] Loading existing global PCA from: {basis_global}")
            V0 = torch.load(basis_global, map_location="cpu")
            return V0, None, None
        return None, None, None

def _map_imagenet_labels_to_basis_ids(y_chunk: torch.Tensor, label2basis: dict):
    """Robustly map y_chunk (shape [B] or [B,1] or [B,1000]) -> basis_id tensor [B]."""
    if y_chunk is None:
        return None
    if y_chunk.ndim > 1:
        y_idx = y_chunk.argmax(dim=1)
    else:
        y_idx = y_chunk
    return torch.as_tensor(
        [label2basis[int(y.item())] for y in y_idx],
        device=y_idx.device,
        dtype=torch.long,
    )

# ---------------- Teacher clustered PCA loader ----------------
@torch.no_grad()
def convert_Q_to_V(Q_2d: torch.Tensor, res: int):
    """
    Q_2d: [d, r]  where d=res*res (4096), r=V_dim (1024)
    Return V: [r, res, res]
    """
    d, r = Q_2d.shape
    assert d == res * res, f"Q has d={d} but expected res*res={res*res}"
    V = Q_2d.T.reshape(r, res, res).contiguous()
    V = V / (V.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)).view(r, 1, 1)
    return V

@torch.no_grad()
def load_teacher_clustered_basis(path: str, device: torch.device, res: int):
    """
    Teacher file structure (your screenshot):
      bases['Q'] shape: [K, d, r] = [60, 4096, 1024]
      bases['class_to_cluster'] shape: [1000]
    We return:
      V_by_label: dict[k] -> [r,res,res]
      label2basis: dict[y] -> k
      V0: arbitrary V for fallback/printing
    """
    bases = torch.load(path, map_location="cpu")
    assert isinstance(bases, dict), "teacher basis must be a dict"

    assert "Q" in bases and "class_to_cluster" in bases, f"Missing keys in teacher basis. keys={list(bases.keys())}"
    Q_all = bases["Q"]  # [K, d, r]
    class_to_cluster = bases["class_to_cluster"]  # [1000]

    assert Q_all.ndim == 3, f"bases['Q'] must be [K,d,r], got {Q_all.shape}"
    K = Q_all.shape[0]
    assert len(class_to_cluster) == 1000, f"class_to_cluster should be length 1000, got {len(class_to_cluster)}"

    V_by_label = {}
    for k in range(K):
        V_by_label[k] = convert_Q_to_V(Q_all[k], res=res).to(device)

    label2basis = {i: int(class_to_cluster[i].item()) for i in range(len(class_to_cluster))}
    # quick sanity
    mn, mx = min(label2basis.values()), max(label2basis.values())
    print(f"[TeacherBasis] Loaded K={K} clusters. label2basis range: [{mn}, {mx}]")
    assert mn >= 0 and mx < K, f"class_to_cluster values out of range: [{mn},{mx}] vs K={K}"

    V0 = next(iter(V_by_label.values()))
    return V0, V_by_label, label2basis

# ---------------- Model builder ----------------
def build_model_from_gonly(dataset, V_dim, res, device, out, outdir: pathlib.Path,
                          dl_for_pca=None, num_classes=None, T=6400.0, K=32,
                          pca_batches=200, per_label_pca=False, num_banks=32,
                          reuse_saved_pca=True,
                          use_teacher_basis=False,
                          teacher_basis_path=None):
    """Load pretrained g-iso weights and initialize g-ani"""
    if dataset == "imagenet":
        gonly_path = os.path.join(out, dataset, f"finetuned-g-iso-32-rampup-10000.pkl")
    else:
        gonly_path = os.path.join(out, dataset, f"finetuned-g-iso-rampup-10000-ema.pkl")

    print(f"Loading g-only initialization from {gonly_path}")
    with dnnlib.util.open_url(gonly_path, "rb") as f:
        ckpt = pickle.load(f)

    # --------- BASIS: teacher clustered PCA (highest priority if enabled) ----------
    V0 = None
    V_by_label = None
    label2basis = None

    if (dataset == "imagenet") and use_teacher_basis:
        assert teacher_basis_path is not None, "--teacher_basis_path is required when --use_teacher_basis"
        print(f"[Basis] Using TEACHER clustered PCA: {teacher_basis_path}")
        V0, V_by_label, label2basis = load_teacher_clustered_basis(teacher_basis_path, device=device, res=res)

    # --------- BASIS: try load saved PCA artifacts ----------
    if V0 is None and reuse_saved_pca:
        V0_cpu, V_by_label_cpu, label2basis_loaded = _try_load_saved_pca(outdir, per_label_pca=per_label_pca)
        if V0_cpu is not None:
            V0 = V0_cpu.to(device)
            if V_by_label_cpu is not None:
                V_by_label = {k: v.to(device) for k, v in V_by_label_cpu.items()}
            label2basis = label2basis_loaded

    # safety if per_label_pca bank mismatch
    if per_label_pca and (V_by_label is not None) and (label2basis is not None):
        nb = len(V_by_label)
        max_v = max(label2basis.values())
        if max_v >= nb:
            print(f"[PCA][WARN] label2basis values go up to {max_v} but V_by_label has {nb} banks. Rebuilding label2basis with mod.")
            label2basis = {y: (y % nb) for y in range(num_classes)}

    # --------- BASIS: compute PCA if still missing ----------
    if V0 is None:
        if per_label_pca and (num_classes is not None) and (dl_for_pca is not None):
            V_by_label_cpu, label2basis = build_V_by_label_pca(
                dl_for_pca, num_classes=num_classes, k=V_dim, d=res, device=device,
                max_batches=pca_batches, dataset=dataset, num_banks=num_banks
            )
            V_by_label = {k: v.to(device) for k, v in V_by_label_cpu.items()}
            V0 = next(iter(V_by_label.values()))
        else:
            assert dl_for_pca is not None, "Need dl_for_pca to compute PCA basis"
            V0 = compute_pca_basis_from_loader(
                dl_for_pca, k=V_dim, d=res, device=device, max_batches=pca_batches, label=None, dataset=dataset
            )

    # --------- base nets ----------
    ema_base = ckpt["ema"].to(device)
    raw_base = ckpt["model"].to(device)
    g_fn = ckpt["g"].to(device)

    ema = ANI_absM_Precond_Flow_Net_basis_v3(ema_base).to(device)
    raw = ANI_absM_Precond_Flow_Net_basis_v3(raw_base).to(device)
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

    print("Model built: g_fn/h_fn initialized (from g-iso)")
    return raw, ema, g_fn, h_fn, V0, V_by_label, label2basis

# ---------------- Train ----------------
def train(opt):
    local_rank, device = setup_ddp()

    # only rank0 prints; others silence
    if not is_main():
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    else:
        sys.stdout = DualWriter(sys.stdout, sys.stderr)

    outdir = pathlib.Path(opt.out) / opt.dataset
    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoints to: {outdir}")
        print("Learning rate:", opt.lr, opt.glr)
        print(f"[DDP] world_size={get_world()} local_rank={local_rank}")

    # data
    dl, V_dim, res, num_classes = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)
    dl_pca, _, _, _ = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)

    # model
    model, ema, g_fn, h_fn, V0, V_by_label, label2basis = build_model_from_gonly(
        opt.dataset, V_dim, res, device, opt.out,
        outdir=outdir,
        dl_for_pca=dl_pca,
        num_classes=num_classes,
        pca_batches=opt.pca_batches,
        per_label_pca=opt.per_label_pca,
        num_banks=opt.num_banks,
        reuse_saved_pca=opt.reuse_saved_pca,
        use_teacher_basis=opt.use_teacher_basis,
        teacher_basis_path=opt.teacher_basis_path,
    )

    # broadcast initial weights so all ranks start identical
    barrier(device_ids=[local_rank])
    broadcast_module_params(model, src=0)
    broadcast_module_params(ema, src=0)
    broadcast_module_params(g_fn, src=0)
    broadcast_module_params(h_fn, src=0)
    barrier(device_ids=[local_rank])

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    # Save PCA only if not exists OR user wants overwrite
    if opt.save_pca:
        if V_by_label is not None and label2basis is not None:
            torch.save({k: v.detach().cpu() for k, v in V_by_label.items()}, outdir / "basis_by_label.pt")
            with open(outdir / "label2basis.pkl", "wb") as f:
                pickle.dump(label2basis, f)
        else:
            torch.save(V0.detach().cpu(), outdir / "basis_global.pt")
        print("Saved basis artifacts.")
    else:
        print("[Basis] save_pca disabled; not writing basis files.")

    opt_model = torch.optim.Adam(ddp_model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_sched = torch.optim.Adam(list(g_fn.parameters()) + list(h_fn.parameters()),
                                 lr=opt.glr, betas=(0.9, 0.999), eps=1e-8)

    scaler = None
    class DummyCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    def autocast_ctx():
        return DummyCtx()

    T = 6400.0
    total_nimg = opt.kimg * 1000
    lr_rampup_kimg = opt.lr_rampup_kimg
    ema_halflife_kimg = 50000
    ema_rampup_ratio = opt.ema_rampup_ratio

    global_batch = opt.batch
    world = get_world()
    assert global_batch % world == 0, f"--batch (global) must be divisible by world_size={world}"
    per_rank_batch = global_batch // world
    steps_total = math.ceil(total_nimg / global_batch)

    pbar = tqdm.tqdm(total=total_nimg, unit="img", dynamic_ncols=True) if is_main() else None
    loss_accum = 0.0

    # quick sanity for teacher basis
    if opt.dataset == "imagenet":
        # assert V_by_label is not None and label2basis is not None, \
        #     "For ImageNet training with basis-v3, you need V_by_label and label2basis."
        nb = len(V_by_label)
        print(f"[Sanity] V_by_label banks={nb}, label2basis range=[{min(label2basis.values())},{max(label2basis.values())}]")

    for step in range(steps_total):
        seen = step * global_batch

        imgs, lbls = next(dl)
        B = imgs.shape[0]
        assert B == global_batch, f"Loader returned batch {B}, expected global_batch {global_batch}"

        # shard by rank
        start = get_rank() * per_rank_batch
        end = (get_rank() + 1) * per_rank_batch
        imgs = imgs[start:end]
        lbls = None if lbls is None else lbls[start:end]

        imgs = imgs.to(device, non_blocking=True)
        if opt.dataset == "cifar10":
            imgs = imgs * 2 - 1
            lbls = None if lbls is None else lbls.to(device, non_blocking=True)
        elif opt.dataset == "imagenet":
            lbls = None if lbls is None else lbls.to(device, non_blocking=True)
        else:
            lbls = None

        opt_model.zero_grad(set_to_none=True)
        opt_sched.zero_grad(set_to_none=True)

        chunk = max(1, per_rank_batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            if opt.dataset == "imagenet":
                basis_chunk = _map_imagenet_labels_to_basis_ids(y_chunk, label2basis)  # [B] long
                # extra sanity
                assert basis_chunk.min().item() >= 0
                assert basis_chunk.max().item() < len(V_by_label)
                y_chunk = y_chunk.to(device=device, dtype=torch.long)

                with autocast_ctx():
                    loss = flow_matching_energy_debug_basis_v3(
                        ddp_model, x_chunk,
                        labels_id=basis_chunk,
                        labels=y_chunk,
                        g_fn=g_fn, h_fn=h_fn,
                        dataset=opt.dataset,
                        T=T,
                        V=V_by_label,
                    )
            else:
                # non-imagenet: keep original behavior (global V0 only)
                loss = flow_matching_energy_debug_basis_v3(
                    ddp_model, x_chunk,
                    labels_id=None,
                    labels=None,
                    g_fn=g_fn, h_fn=h_fn,
                    dataset=opt.dataset,
                    T=T,
                    V=None,  # or pass global if your v3 supports it
                )

            loss_accum += loss.item()
            (loss / x_chunk.shape[0]).backward()

        # sanitize grads
        for p in list(ddp_model.parameters()) + list(g_fn.parameters()) + list(h_fn.parameters()):
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

        allreduce_grads(g_fn)
        allreduce_grads(h_fn)

        if step > 0:
            opt_model.step()
            opt_sched.step()

        # EMA
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, seen * ema_rampup_ratio)
        ema_beta = 0.5 ** (opt.batch / max(ema_halflife_nimg, 1e-8))
        if is_main():
            with torch.no_grad():
                for p_e, p_m in zip(ema.parameters(), ddp_model.parameters()):
                    p_e.copy_(p_m.detach().lerp(p_e, ema_beta))

        broadcast_module_params(ema, src=0)

        seen_after = min(seen + global_batch, total_nimg)
        if is_main():
            pbar.update(min(global_batch, total_nimg - seen))

        if is_main() and (step + 1) % 10 == 0:
            avg = loss_accum / 10
            print(f"[{opt.dataset}] Seen {seen_after}/{total_nimg} | Avg loss {avg:.3e} | lr_scale {lr_scale:.3f}")
            loss_accum = 0.0

        if (step + 1) % 100 == 0:
            barrier(device_ids=[local_rank])
            if is_main():
                if opt.keep_all_ckpt:
                    ckpt_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-{ema_halflife_kimg}-basis-v3-newpca-ddp-ckpt-{(step+1):05d}.pkl"
                else:
                    ckpt_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-{ema_halflife_kimg}-basis-v3-newpca-ddp.pkl"

                with open(ckpt_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": ddp_model.module.cpu(),
                            "ema": ema.cpu(),
                            "g_fn": g_fn.cpu(),
                            "h_fn": h_fn.cpu(),
                        },
                        f
                    )
                print(f"Saved: {ckpt_path}")

                # move back
                ddp_model.module.to(device)
                ema.to(device)
                g_fn.to(device)
                h_fn.to(device)

            barrier(device_ids=[local_rank])

        barrier(device_ids=[local_rank])

    barrier(device_ids=[local_rank])
    if is_main():
        final_path = outdir / f"finetuned-g-ani-rampup-{lr_rampup_kimg}-ema-{ema_halflife_kimg}-basis-v3-newpca-ddp.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(
                {
                    "model": ddp_model.module.cpu(),
                    "ema": ema.cpu(),
                    "g_fn": g_fn.cpu(),
                    "h_fn": h_fn.cpu(),
                },
                f
            )
        print(f"Training finished. Final checkpoint: {final_path}")


# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet", choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--out", default="finetune_initial")
    parser.add_argument("--batch", type=int, default=516)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--glr", type=float, default=1e-4)
    parser.add_argument("--kimg", type=int, default=1200)
    parser.add_argument("--grad_accum", type=int, default=11)
    parser.add_argument('--lr_rampup_kimg', type=int, default=10000)
    parser.add_argument("--ema_rampup_ratio", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument('--keep_all_ckpt', action='store_true')

    # Old PCA options (kept as fallback)
    parser.add_argument("--per_label_pca", action="store_true",
                        help="If set, compute one PCA basis per label bank and use it in the loss (fallback path).")
    parser.add_argument("--pca_batches", type=int, default=200,
                        help="How many batches to use for PCA estimation (only used if PCA not loaded).")
    parser.add_argument("--num_banks", type=int, default=32,
                        help="Label-aware basis bank size for ImageNet when --per_label_pca is set.")
    parser.add_argument("--reuse_saved_pca", action="store_true",
                        help="If set, try loading basis_*.pt and label2basis.pkl from out/dataset before recomputing PCA.")
    parser.add_argument("--save_pca", action="store_true",
                        help="If set, save basis artifacts to out/dataset. (Recommended first run.)")

    # NEW: teacher clustered basis
    parser.add_argument("--use_teacher_basis", action="store_true",
                        help="If set (ImageNet only), load teacher clustered PCA basis: Q + class_to_cluster.")
    parser.add_argument("--teacher_basis_path", type=str, default="/data/finetune_initial/imagenet/clustered_pca_via_classPCA128_K60_r1024_bestseed1.pt",
                        help="Path to teacher clustered PCA .pt file (contains keys: Q, class_to_cluster, ...).")

    args = parser.parse_args()
    train(args)
