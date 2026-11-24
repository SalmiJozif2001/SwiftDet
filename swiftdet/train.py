import argparse
import copy
import csv
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from swiftdet.data.voc_dataset import VOCDataset, collate_fn
from swiftdet.losses.gfl_simota import SwiftDetLoss
from swiftdet.models.swiftdet import SwiftDet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach().to(v.device) * (1.0 - d))


def build_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def lr_at(epoch, tcfg, total_epochs):
    lr0, warm = tcfg["lr0"], tcfg["warmup_epochs"]
    if epoch < warm:
        return lr0 * float(epoch + 1) / warm
    progress = (epoch - warm) / max(total_epochs - warm, 1)
    return lr0 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def run_validation(ema_model, voc_root, img_size, device, max_images):
    from swiftdet.eval_voc import evaluate_fn

    ema_model.eval()

    def predict(x):
        t = torch.from_numpy(x).to(device)
        return ema_model(t).float().cpu().numpy()

    return evaluate_fn(predict, voc_root, img_size=img_size, max_images=max_images, verbose=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="swiftdet/configs/voc_swiftdet.yaml")
    p.add_argument("--data-root", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-dir", type=str, default="runs/swiftdet_voc480")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--assigner", type=str, default="")
    p.add_argument("--subset-voc07", action="store_true")
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument("--val-every", type=int, default=-1)
    p.add_argument("--time-limit-hours", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["train"]
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(tcfg["seed"])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    amp_on = bool(tcfg["amp_fp16"]) and use_cuda
    use_cl = bool(tcfg["channels_last"]) and use_cuda
    img_size = cfg["model"]["input_size"][0]
    data_root = args.data_root or cfg["data"]["root"]
    batch_size = args.batch_size if args.batch_size > 0 else tcfg["batch_size"]
    workers = args.workers if args.workers >= 0 else cfg["data"]["num_workers"]
    val_every = args.val_every if args.val_every >= 0 else tcfg.get("val_every", 0)
    val_images = tcfg.get("val_images", 500)
    train_sets = [["2007", "trainval"]] if args.subset_voc07 else cfg["data"]["train_sets"]
    assigner = args.assigner or tcfg.get("assigner", "simota")
    total_epochs = args.epochs if args.epochs > 0 else tcfg["epochs"]

    print(f"[Setup] device={device} amp={amp_on} channels_last={use_cl} batch={batch_size} "
          f"workers={workers} assigner={assigner} epochs={total_epochs}", flush=True)

    train_ds = VOCDataset(
        data_root, sets=train_sets, img_size=img_size, train=True,
        mosaic_prob=tcfg["mosaic_prob"], mixup_prob=tcfg["mixup_prob"],
    )
    print(f"[Data] train images = {len(train_ds)}", flush=True)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=use_cuda, persistent_workers=False, collate_fn=collate_fn, drop_last=True,
    )

    model = SwiftDet(
        num_classes=cfg["model"]["num_classes"],
        reg_max=cfg["model"]["reg_max"],
        neck_channels=cfg["model"]["neck_channels"],
    ).to(device)
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] training-graph params = {n_params:,}", flush=True)

    criterion = SwiftDetLoss(
        num_classes=cfg["model"]["num_classes"],
        reg_max=cfg["model"]["reg_max"],
        strides=tuple(cfg["model"]["strides"]),
        img_size=img_size,
        vfl_w=tcfg["loss_weights"]["vfl"],
        giou_w=tcfg["loss_weights"]["giou"],
        dfl_w=tcfg["loss_weights"]["dfl"],
        assigner=assigner,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=tcfg["lr0"], momentum=tcfg["momentum"],
        weight_decay=tcfg["weight_decay"], nesterov=True,
    )
    scaler = build_scaler(amp_on)
    ema = ModelEMA(model, decay=tcfg["ema_decay"])
    grad_clip = float(tcfg.get("grad_clip", 10.0))

    def prep(imgs):
        imgs = imgs.to(device, non_blocking=True)
        if use_cl:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        return imgs

    def forward_loss(imgs, targets):
        # Only the network forward runs under autocast; the loss is always computed in FP32.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            preds = model(imgs)
        return criterion(preds.float(), targets)

    # ------------------------------------------------------------------ overfit sanity check
    if args.overfit_one_batch:
        imgs, targets = next(iter(loader))
        imgs = prep(imgs)
        model.train()
        first_loss, last_loss = None, None
        for step in range(1, 201):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = forward_loss(imgs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            last_loss = float(loss.item())
            if first_loss is None:
                first_loss = last_loss
            if step % 20 == 0 or step == 1:
                print(f"[Overfit Check] step={step:3d} loss={last_loss:.4f} "
                      f"(vfl={float(parts['vfl']):.3f}, giou={float(parts['giou']):.3f}, "
                      f"dfl={float(parts['dfl']):.3f})", flush=True)
        verdict = "PASS" if (last_loss is not None and last_loss < 0.6 * first_loss) else "WARN (loss did not drop enough)"
        print(f"[Overfit Check] {verdict}: first={first_loss:.4f} last={last_loss:.4f}", flush=True)
        return

    # ------------------------------------------------------------------ resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        ema.updates = int(ckpt.get("ema_updates", start_epoch * max(len(loader), 1)))
        print(f"[Resume] from {args.resume} -> starting at epoch {start_epoch}", flush=True)
    elif args.resume:
        print(f"[Resume] file not found: {args.resume} (starting from scratch)", flush=True)

    log_path = os.path.join(args.save_dir, "train_log.csv")
    need_header = (not os.path.isfile(log_path)) or os.path.getsize(log_path) == 0
    log_csv = open(log_path, "a", newline="")
    writer = csv.writer(log_csv)
    if need_header:
        writer.writerow(["epoch", "lr", "loss", "vfl", "giou", "dfl", "epoch_sec", "val_ap50", "val_map"])
        log_csv.flush()

    if start_epoch >= total_epochs:
        print("TRAINING_COMPLETE (checkpoint already at the final epoch)", flush=True)
        log_csv.close()
        return

    session_t0 = time.time()
    limit_sec = args.time_limit_hours * 3600.0 if args.time_limit_hours > 0 else None
    stopped_early = False
    n_batches_total = max(len(loader), 1)

    for epoch in range(start_epoch, total_epochs):
        if epoch >= total_epochs - tcfg["no_aug_epochs"]:
            train_ds.mosaic_prob = 0.0
            train_ds.mixup_prob = 0.0

        lr = lr_at(epoch, tcfg, total_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        t0 = time.time()
        run_loss = run_vfl = run_giou = run_dfl = 0.0
        n_ok, consecutive_bad = 0, 0
        for imgs, targets in loader:
            imgs = prep(imgs)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = forward_loss(imgs, targets)
            if not torch.isfinite(loss):
                consecutive_bad += 1
                print(f"[Warn] non-finite loss at epoch {epoch}, skipping batch ({consecutive_bad})", flush=True)
                if consecutive_bad >= 20:
                    raise RuntimeError("Too many consecutive non-finite losses; lower lr0 or switch assigner to atss.")
                continue
            consecutive_bad = 0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            run_loss += float(loss.item())
            run_vfl += float(parts["vfl"])
            run_giou += float(parts["giou"])
            run_dfl += float(parts["dfl"])
            n_ok += 1

        dt = time.time() - t0
        n_ok = max(n_ok, 1)
        avg_loss, avg_vfl, avg_giou, avg_dfl = run_loss / n_ok, run_vfl / n_ok, run_giou / n_ok, run_dfl / n_ok

        val_ap50, val_map = "", ""
        is_last_epoch = epoch == total_epochs - 1
        if val_every > 0 and ((epoch + 1) % val_every == 0 or is_last_epoch):
            try:
                ap50, mp = run_validation(ema.ema, data_root, img_size, device, val_images)
                val_ap50, val_map = f"{ap50:.4f}", f"{mp:.4f}"
                print(f"[Val] epoch {epoch:03d} | AP50={ap50 * 100:.2f}% | mAP50-95={mp * 100:.2f}% "
                      f"(EMA, first {val_images} imgs of VOC07 test)", flush=True)
            except Exception as e:  # validation must never kill a long training run
                print(f"[Val] skipped due to error: {e}", flush=True)

        writer.writerow([epoch, f"{lr:.6f}", f"{avg_loss:.4f}", f"{avg_vfl:.4f}", f"{avg_giou:.4f}",
                         f"{avg_dfl:.4f}", f"{dt:.1f}", val_ap50, val_map])
        log_csv.flush()
        print(f"Epoch {epoch:03d}/{total_epochs} | lr={lr:.5f} | loss={avg_loss:.4f} | "
              f"vfl={avg_vfl:.3f} giou={avg_giou:.3f} dfl={avg_dfl:.3f} | time={dt:.1f}s", flush=True)

        atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, os.path.join(args.save_dir, "last.pt"))
        if (epoch + 1) % 25 == 0 or is_last_epoch:
            atomic_save(ema.ema.state_dict(), os.path.join(args.save_dir, f"ema_ep{epoch:03d}.pt"))

        elapsed = time.time() - session_t0
        if limit_sec is not None and not is_last_epoch and elapsed + dt * 1.15 > limit_sec:
            print(f"TIME_LIMIT_REACHED after epoch {epoch}: elapsed {elapsed / 3600.0:.2f} h. "
                  f"Checkpoint saved to last.pt - resume in the next session.", flush=True)
            stopped_early = True
            break

    log_csv.close()
    if not stopped_early:
        print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
