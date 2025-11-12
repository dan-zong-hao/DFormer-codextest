import argparse
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.builder import EncoderDecoder
from utils_1 import ISPRS_dataset, N_CLASSES, test_ids, BATCH_SIZE


IGNORE_INDEX = 255


def build_config(num_classes: int) -> SimpleNamespace:
    """Return a minimal config namespace compatible with :class:`EncoderDecoder`."""
    return SimpleNamespace(
        backbone="DFormerv2_L",
        decoder="ham",
        decoder_embed_dim=768,
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

    The NYU checkpoint was trained with 40 classes, so we ignore mismatched
    decoder weights when adapting to the ISPRS label space by default.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
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

    for step, batch in enumerate(loader):
        if max_iters is not None and step >= max_iters:
            break
        rgb = batch["data"].to(device, non_blocking=True)
        depth = batch["modal_x"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)

        if autocast is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(rgb, depth)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, target)
        else:
            logits = model(rgb, depth)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, target)

        total_loss += loss.detach() * torch.sum(target != IGNORE_INDEX)
        total_pixels += torch.sum(target != IGNORE_INDEX).item()

        preds = logits.argmax(dim=1)
        mask = target != IGNORE_INDEX
        if mask.any():
            hist = torch.bincount(
                (target[mask] * num_classes + preds[mask]).view(-1),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DFormerv2_L on ISPRS dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to DFormerv2_Large_NYU.pth")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--max-iters", type=int, default=None, help="Optional cap on evaluation iterations")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision during evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = build_config(N_CLASSES)
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    load_checkpoint(model, args.checkpoint, strict=False)
    model.to(device)

    ids = test_ids if args.split == "test" else train_ids
    dataset = ISPRS_dataset(ids, cache=True, augmentation=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics = evaluate(
        model,
        loader,
        device=device,
        num_classes=N_CLASSES,
        amp=not args.no_amp,
        max_iters=args.max_iters,
    )

    print("Evaluation metrics:\n"
          f"  mIoU: {metrics['miou']:.4f}\n"
          f"  Pixel Acc: {metrics['pixel_acc']:.4f}\n"
          f"  Mean Acc: {metrics['mean_acc']:.4f}\n"
          f"  Loss: {metrics['loss']:.4f}")
    print("Per-class IoU:")
    for idx, score in enumerate(metrics["iou"]):
        print(f"  Class {idx}: {score:.4f}")


if __name__ == "__main__":
    from utils_1 import train_ids  # defer to avoid circular import at module load

    main()
