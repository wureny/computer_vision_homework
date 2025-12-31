"""
Training entry for the meme template classifier.

Key steps:
- Load ImageFolder dataset from `data_dir/train` and `data_dir/val`
- Build MobileNetV2 (optionally ImageNet-pretrained) and replace the classifier head
- (Optional) freeze the backbone for a lightweight baseline
- Train with cross-entropy; save best validation checkpoint and copy it to `pretrained/`
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vision_utils import build_mobilenetv2, save_checkpoint, save_label_map


def make_transforms(train: bool) -> transforms.Compose:
    # Standard ImageNet-style preprocessing; simple augmentations for small datasets.
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 meme template classifier.")
    parser.add_argument("--data_dir", type=str, default="data", help="Contains train/val folders.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze feature extractor.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument(
        "--weights_out",
        type=str,
        default="pretrained/mobilenetv2_meme_best.pt",
        help="Where to write the best checkpoint (for TA testing).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected folders not found: {train_dir} and {val_dir}\n"
            "Tip: run prepare_dataset.py or create data/train and data/val manually."
        )

    train_ds = datasets.ImageFolder(train_dir.as_posix(), transform=make_transforms(train=True))
    val_ds = datasets.ImageFolder(val_dir.as_posix(), transform=make_transforms(train=False))
    if len(train_ds.classes) < 2:
        raise SystemExit("Need at least 2 classes to train.")

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Transfer learning: start from ImageNet-pretrained MobileNetV2, replace final classifier.
    model = build_mobilenetv2(num_classes=len(train_ds.classes), pretrained=True)
    if args.freeze_backbone:
        # Baseline: only train the classifier head.
        for p in model.features.parameters():
            p.requires_grad = False
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_label_map(run_dir / "class_to_idx.json", train_ds.class_to_idx)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            pbar.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device=device)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                class_to_idx=train_ds.class_to_idx,
                meta={"epoch": epoch, "val_acc": val_acc},
            )

    # Copy best checkpoint into a stable path for TA testing / inference scripts.
    src = run_dir / "best.pt"
    dst = Path(args.weights_out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print("Training finished.")
    print("Best val acc:", best_val_acc)
    print("Saved:", dst)


if __name__ == "__main__":
    main()
