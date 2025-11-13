import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from models.builder import EncoderDecoder
from utils_1 import (
    BATCH_SIZE,
    ISPRS_dataset,
    N_CLASSES,
    WEIGHTS,
    train_ids,
    test_ids,
)

IGNORE_INDEX = 255


def build_config(num_classes: int) -> argparse.Namespace:
    """Return a config namespace aligned with the NYU encoder-decoder checkpoint."""
    return argparse.Namespace(
        backbone="DFormerv2_L",
        decoder="ham",
        decoder_embed_dim=1024,
        num_classes=num_classes,
        drop_path_rate=0.3,
        bn_eps=1e-5,
        bn_momentum=0.1,
        aux_rate=0.0,
        pretrained_model=None,
        background=IGNORE_INDEX,
    )


def load_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load weights from ``ckpt_path`` into ``model``.

    Parameters whose shapes differ from the current module are discarded so the
    pretrained encoder weights can still be reused even when the segmentation
    head has a different number of classes.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            payload = state.get(key)
            if isinstance(payload, dict):
                state = payload
                break

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state.items():
        target = model_state.get(key)
        if target is None or target.shape != value.shape:
            skipped.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=strict)
    if skipped:
        print(f"Skipped incompatible keys ({len(skipped)}): {skipped}")
    if missing:
        print(f"Missing keys after load ({len(missing)}): {missing}")
    if unexpected:
        print(f"Unexpected keys after load ({len(unexpected)}): {unexpected}")


def freeze_backbone(model: EncoderDecoder) -> None:
    """Freeze backbone parameters and set them to evaluation mode."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()
    for module in model.backbone.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def build_dataloaders(
    batch_size: int,
    num_workers: int,
    cache: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ISPRS_dataset(train_ids, cache=cache, augmentation=True)
    val_dataset = ISPRS_dataset(test_ids, cache=cache, augmentation=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(
    model: EncoderDecoder,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp: bool = False,
    max_iters: Optional[int] = None,
) -> dict:
    model.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_pixels = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    autocast = torch.autocast if amp and device.type == "cuda" else None

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_iters is not None and step >= max_iters:
                break
            rgb = batch["data"].to(device, non_blocking=True)
            depth = batch["modal_x"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            if autocast is not None:
                with autocast(device_type=device.type, dtype=torch.float16):
                    logits = model.encode_decode(rgb, depth)
                    loss = criterion(logits, target)
            else:
                logits = model.encode_decode(rgb, depth)
                loss = criterion(logits, target)

            valid_mask = target != IGNORE_INDEX
            total_loss += loss.detach() * valid_mask.sum()
            total_pixels += valid_mask.sum().item()

            preds = logits.argmax(dim=1)
            if valid_mask.any():
                hist = torch.bincount(
                    (target[valid_mask] * num_classes + preds[valid_mask]).view(-1),
                    minlength=num_classes * num_classes,
                )
                conf_mat += hist.view(num_classes, num_classes)

    conf_cpu = conf_mat.cpu().float()
    intersection = torch.diag(conf_cpu)
    gt_sum = conf_cpu.sum(dim=1)
    pred_sum = conf_cpu.sum(dim=0)
    union = gt_sum + pred_sum - intersection
    valid = union > 0
    iou = torch.zeros(num_classes)
    iou[valid] = intersection[valid] / union[valid]
    miou = iou[valid].mean().item() if valid.any() else 0.0
    pixel_acc = intersection.sum().item() / union.sum().item() if union.sum() > 0 else 0.0
    mean_acc = (intersection / gt_sum.clamp_min(1)).nanmean().item()
    mean_loss = (total_loss / total_pixels).item() if total_pixels > 0 else 0.0

    return {
        "iou": iou.tolist(),
        "miou": miou,
        "pixel_acc": pixel_acc,
        "mean_acc": mean_acc,
        "loss": mean_loss,
    }


def save_checkpoint(model: EncoderDecoder, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train_one_epoch(
    model: EncoderDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    log_interval: int,
) -> float:
    model.train()
    model.backbone.eval()

    running_loss = 0.0
    total_pixels = 0
    autocast = torch.autocast if scaler is not None else None

    for step, batch in enumerate(loader):
        rgb = batch["data"].to(device, non_blocking=True)
        depth = batch["modal_x"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        valid_mask = target != IGNORE_INDEX

        if scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model.encode_decode(rgb, depth)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.encode_decode(rgb, depth)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        batch_pixels = valid_mask.sum().item()
        total_pixels += batch_pixels
        running_loss += loss.detach().item() * batch_pixels

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / max(total_pixels, 1)
            print(f"Epoch [{epoch}] Step [{step + 1}/{len(loader)}] Loss: {avg_loss:.6f}")

    epoch_loss = running_loss / max(total_pixels, 1)
    return epoch_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DFormerv2 decoder on ISPRS")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/NYU/DFormerv2_Large_NYU.pth"), help="Path to DFormerv2_Large_NYU.pth")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for decoder parameters")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--log-interval", type=int, default=50, help="Iterations between logging steps")
    parser.add_argument("--val-max-iters", type=int, default=None, help="Optional cap on validation iterations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ftmodel_v"),
        help="Directory to store fine-tuned checkpoints",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Store checkpoint with the best validation mIoU in addition to latest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = build_config(N_CLASSES)
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    load_checkpoint(model, args.checkpoint, strict=False)
    freeze_backbone(model)
    model.to(device)

    train_loader, val_loader = build_dataloaders(args.batch_size, args.num_workers)

    trainable_params = list(model.decode_head.parameters())
    if model.aux_head is not None:
        trainable_params += list(model.aux_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    class_weights = WEIGHTS.clone().float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler()

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            criterion,
            epoch,
            max(args.log_interval, 1),
        )
        print(f"Epoch {epoch} training loss: {train_loss:.6f}")

        metrics = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=N_CLASSES,
            amp=args.amp,
            max_iters=args.val_max_iters,
        )

        print(
            "Validation metrics:\n"
            f"  mIoU: {metrics['miou']:.4f}\n"
            f"  Pixel Acc: {metrics['pixel_acc']:.4f}\n"
            f"  Mean Acc: {metrics['mean_acc']:.4f}\n"
            f"  Loss: {metrics['loss']:.6f}"
        )

        latest_path = args.output_dir / "decoder_finetune_latest.pth"
        save_checkpoint(model, optimizer, epoch, latest_path)

        if args.save_best and metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            best_path = args.output_dir / "decoder_finetune_best.pth"
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"New best mIoU {best_miou:.4f}, checkpoint saved to {best_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()