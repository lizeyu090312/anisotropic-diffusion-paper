import argparse, pathlib, pickle, torch, tqdm, dnnlib, math, copy, sys, os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common_utils import GFn, compute_DCT_basis, flow_matching_energy_debug_condition, ANI_absM_Precond_Flow_Net
from data_loader import afhqv2_loader, ffhq_loader, cifar10_loader, imagenet_loader


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
    # [FIX] specify device_ids to avoid "devices used unknown" warning & potential hang.
    if ddp_is_initialized():
        if device_ids is None:
            dist.barrier()
        else:
            dist.barrier(device_ids=device_ids)

def broadcast_module_params(module, src=0):
    """Broadcast all parameters/buffers of a module from src to all ranks."""
    if not ddp_is_initialized():
        return
    # IMPORTANT: order must match across ranks; state_dict() order is deterministic for same module
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src=src)

def allreduce_grads(module):
    """
    All-reduce (sum) grads, then average by world size.

    [FIX] Critical: NEVER skip parameters with grad=None.
    If some rank didn't use this module in forward, grad can be None,
    but other ranks will have a real grad -> skipping causes different
    collective call counts -> NCCL deadlock.
    """
    if not ddp_is_initialized():
        return
    world = get_world()
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            # [FIX] create a zero grad so every rank participates in allreduce for every param
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)

def gather_used_labels(used_labels_set):
    """Gather per-rank used label set -> global union set (python object gather)."""
    if not ddp_is_initialized():
        return set(used_labels_set)
    gathered = [None for _ in range(get_world())]
    dist.all_gather_object(gathered, sorted(list(used_labels_set)))
    out = set()
    for lst in gathered:
        for x in lst:
            out.add(int(x))
    return out


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
def build_model_from_gonly(dataset, V_dim, res, device, out, num_classes):
    gonly_path = os.path.join(out, dataset, "finetuned-g-iso-condition-debug.pkl")
    if is_main():
        print(f"Loading g initialization from {gonly_path}")

    with dnnlib.util.open_url(gonly_path, "rb") as f:
        ckpt = pickle.load(f)

    dct_V = compute_DCT_basis(k=V_dim, d=res, dtype=torch.float32, device=device)

    # base nets
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

    # label-wise g/h
    g_fns = torch.nn.ModuleDict()
    h_fns = torch.nn.ModuleDict()

    if "g_fns" in ckpt:
        if is_main():
            print("[Init] Using label-wise g_fns from g-iso-condition checkpoint")
        for c in range(num_classes):
            key = str(c)
            if key not in ckpt["g_fns"]:
                raise KeyError(f"Missing g_fns[{key}] in checkpoint")
            g_fns[key] = ckpt["g_fns"][key].to(device)
            h_fns[key] = copy.deepcopy(g_fns[key])  # start isotropic
    elif "g" in ckpt:
        if is_main():
            print("[Init] Using single g and copying to all labels")
        g_base = ckpt["g"].to(device)
        for c in range(num_classes):
            g_fns[str(c)] = copy.deepcopy(g_base).to(device)
            h_fns[str(c)] = copy.deepcopy(g_base).to(device)
    else:
        raise KeyError("Checkpoint must contain either `g_fns` or `g`")

    if is_main():
        print(f"Model built with label-wise g/h (num_classes={num_classes})")

    return raw, ema, g_fns, h_fns, dct_V


# ---------------- Train ----------------
def train(opt):
    local_rank, device = setup_ddp()

    # only rank0 prints to stdout/stderr; other ranks silence to avoid log spam
    if not is_main():
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    else:
        sys.stdout = DualWriter(sys.stdout, sys.stderr)

    # setup output dir
    outdir = pathlib.Path(opt.out) / opt.dataset
    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoints to: {outdir}")
        print("Learning rate:", opt.lr, opt.glr)
        print(f"[DDP] world_size={get_world()} local_rank={local_rank}")

    # loader: IMPORTANT - opt.batch is GLOBAL batch (all GPUs)
    dl, V_dim, res, num_classes = get_dataset_loader(opt.dataset, opt.batch, workers=opt.workers)

    # model
    model, ema, g_fns, h_fns, _ = build_model_from_gonly(opt.dataset, V_dim, res, device, opt.out, num_classes)

    # broadcast initial weights so all ranks start identical (safety)
    barrier(device_ids=[local_rank])
    broadcast_module_params(model, src=0)
    broadcast_module_params(ema, src=0)
    for k in g_fns.keys():
        broadcast_module_params(g_fns[k], src=0)
        broadcast_module_params(h_fns[k], src=0)
    barrier(device_ids=[local_rank])

    # DDP wrapper for the main model
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    # optimizers
    opt_model = torch.optim.Adam(ddp_model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    opt_sched = {
        c: torch.optim.Adam(
            list(g_fns[str(c)].parameters()) + list(h_fns[str(c)].parameters()),
            lr=opt.glr, betas=(0.9, 0.999), eps=1e-8
        )
        for c in range(num_classes)
    }

    # AMP
    use_amp = opt.amp
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        scaler = None
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        autocast = lambda: DummyCtx()

    # training hyperparams
    T = 6400.0
    total_nimg = opt.kimg * 1000  # total images (global count)
    lr_rampup_kimg = 10000
    ema_halflife_kimg = 50000
    ema_rampup_ratio = opt.ema_rampup_ratio

    # fixed #steps to keep all ranks in lockstep
    global_batch = opt.batch
    world = get_world()
    assert global_batch % world == 0, f"--batch (global) must be divisible by world_size={world}"
    per_rank_batch = global_batch // world
    steps_total = math.ceil(total_nimg / global_batch)

    pbar = tqdm.tqdm(total=total_nimg, unit="img", dynamic_ncols=True) if is_main() else None
    loss_accum = 0.0

    for step in range(steps_total):
        seen = step * global_batch

        imgs, lbls = next(dl)

        # shard the batch across ranks
        B = imgs.shape[0]
        assert B == global_batch, f"Loader returned batch {B}, expected global_batch {global_batch}"
        start = get_rank() * per_rank_batch
        end = (get_rank() + 1) * per_rank_batch
        imgs = imgs[start:end]
        lbls = None if lbls is None else lbls[start:end]

        imgs = imgs.to(device, non_blocking=True)
        if opt.dataset == "cifar10":
            imgs = imgs * 2 - 1
            lbls = lbls.to(device, non_blocking=True)
        elif opt.dataset == "imagenet":
            lbls = lbls.to(device, non_blocking=True)
        else:
            lbls = None

        opt_model.zero_grad(set_to_none=True)
        for o in opt_sched.values():
            o.zero_grad(set_to_none=True)

        # grad accumulation chunks (per rank)
        chunk = max(1, per_rank_batch // opt.grad_accum)
        x_chunks = torch.split(imgs, chunk)
        y_chunks = [None] * len(x_chunks) if lbls is None else torch.split(lbls, chunk)

        used_labels = set()

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            if y_chunk is None:
                class_ids = torch.zeros(x_chunk.size(0), dtype=torch.long, device=x_chunk.device)
                y_for_loss = None
            else:
                if y_chunk.ndim == 2:
                    class_ids = y_chunk.argmax(dim=1)
                    y_for_loss = y_chunk
                else:
                    class_ids = y_chunk
                    y_for_loss = y_chunk

            # [FIX] always have a tensor loss so we always backward once per chunk
            # total_loss = torch.zeros([], device=device, dtype=torch.float32)

            with autocast():
                total_loss = flow_matching_energy_debug_condition(
                    ddp_model,
                    x_chunk,
                    y_chunk,
                    g_fns,
                    h_fns,
                    opt.dataset,
                    T=T,
                    tmin=1e-9,
                    class_ids=class_ids
                )
            
            used_labels.update(class_ids.unique().tolist())

            if use_amp:
                scaler.scale(total_loss / opt.grad_accum).backward()
            else:
                (total_loss / opt.grad_accum).backward()

            # logging accumulator (rank0 only)
            loss_accum += float((total_loss.detach() / max(1, x_chunk.shape[0])).item())

        # sanitize grads (local)
        for p in ddp_model.parameters():
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

        # LR schedule uses GLOBAL seen
        ramp = min(seen / max(lr_rampup_kimg * 1000, 1e-8), 1.0)
        decay = 1.0 if seen <= total_nimg // 2 else 0.5 ** ((seen - total_nimg // 2) / (total_nimg // 8))
        lr_scale = ramp * decay
        for g in opt_model.param_groups:
            g["lr"] = opt.lr * lr_scale
        for o in opt_sched.values():
            for pg in o.param_groups:
                pg["lr"] = opt.glr * lr_scale

        # union of used labels
        global_used = gather_used_labels(used_labels)

        # [FIX] allreduce grads for g/h: never skip grad=None
        for lbl in global_used:
            allreduce_grads(g_fns[str(lbl)])
            allreduce_grads(h_fns[str(lbl)])

        # optimizer step
        if use_amp:
            scaler.step(opt_model)
            for lbl in global_used:
                scaler.step(opt_sched[lbl])
            scaler.update()
        else:
            opt_model.step()
            for lbl in global_used:
                opt_sched[lbl].step()

        # EMA update on rank0 then broadcast to all ranks
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, seen * ema_rampup_ratio)
        ema_beta = 0.5 ** (global_batch / max(ema_halflife_nimg, 1e-8))

        if is_main():
            with torch.no_grad():
                for p_e, p_m in zip(ema.parameters(), ddp_model.module.parameters()):
                    p_e.copy_(p_m.detach().lerp(p_e, ema_beta))

        # broadcast EMA params so all ranks keep identical ema
        broadcast_module_params(ema, src=0)

        # progress accounting (global)
        seen_after = min(seen + global_batch, total_nimg)
        if is_main():
            pbar.update(min(global_batch, total_nimg - seen))

        # logging
        if is_main() and (step + 1) % 10 == 0:
            avg = loss_accum / 10
            print(f"[{opt.dataset}] Seen {seen_after}/{total_nimg} | Avg loss {avg:.3e} | lr_scale {lr_scale:.3f}")
            loss_accum = 0.0

        # checkpoint (sync all ranks around save)
        if (step + 1) % 100 == 0:
            barrier(device_ids=[local_rank])
            if is_main():
                if opt.keep_all_ckpt:
                    ckpt_path = outdir / f"finetuned-g-ani-condition-v2-ema-{ema_halflife_kimg}-ddp-ckpt-{(step+1):05d}.pkl"
                else:
                    ckpt_path = outdir / f"finetuned-g-ani-condition-v2-ema-{ema_halflife_kimg}-ddp.pkl"

                with open(ckpt_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": ddp_model.module.cpu(),
                            "ema": ema.cpu(),
                            "g_fns": {k: v.cpu() for k, v in g_fns.items()},
                            "h_fns": {k: v.cpu() for k, v in h_fns.items()},
                        },
                        f
                    )
                print(f"Saved: {ckpt_path}")

                ddp_model.module.to(device)
                ema.to(device)
                for v in g_fns.values(): v.to(device)
                for v in h_fns.values(): v.to(device)
            barrier(device_ids=[local_rank])

        barrier(device_ids=[local_rank])

    # final save
    barrier(device_ids=[local_rank])
    if is_main():
        final_path = outdir / f"finetuned-g-ani-condition-v2-ddp-ema-{ema_halflife_kimg}.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(
                {
                    "model": ddp_model.module.cpu(),
                    "ema": ema.cpu(),
                    "g_fns": {k: v.cpu() for k, v in g_fns.items()},
                    "h_fns": {k: v.cpu() for k, v in h_fns.items()},
                },
                f
            )
        print(f"Training finished. Final checkpoint: {final_path}")
    barrier(device_ids=[local_rank])

    dist.destroy_process_group()


# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "afhqv2", "ffhq", "imagenet"])
    parser.add_argument("--out", default="finetune_initial")
    parser.add_argument("--batch", type=int, default=256, help="GLOBAL batch across all GPUs; must be divisible by world_size")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--glr", type=float, default=1e-4)
    parser.add_argument("--kimg", type=int, default=1200)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--ema_rampup_ratio", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable AMP fp16 autocast + GradScaler")
    parser.add_argument("--keep_all_ckpt", action="store_true")

    args = parser.parse_args()
    train(args)
