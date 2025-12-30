import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_utils import idx_to_class, load_checkpoint


def make_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on data/test.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_checkpoint(args.weights, device=device)
    model = bundle.model
    class_to_idx = bundle.class_to_idx
    id2class = idx_to_class(class_to_idx)

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        raise SystemExit(f"Expected folder not found: {test_dir}")

    test_ds = datasets.ImageFolder(test_dir.as_posix(), transform=make_transforms())
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Map indices used by ImageFolder(test) to indices used by the trained checkpoint.
    if set(test_ds.class_to_idx.keys()) != set(class_to_idx.keys()):
        missing_in_test = sorted(set(class_to_idx.keys()) - set(test_ds.class_to_idx.keys()))
        missing_in_ckpt = sorted(set(test_ds.class_to_idx.keys()) - set(class_to_idx.keys()))
        raise SystemExit(
            "Class names mismatch between checkpoint and test folder.\n"
            f"Missing in test: {missing_in_test}\n"
            f"Missing in checkpoint: {missing_in_ckpt}"
        )
    test_idx_to_ckpt_idx = {test_ds.class_to_idx[c]: class_to_idx[c] for c in test_ds.class_to_idx}

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend([test_idx_to_ckpt_idx[int(i)] for i in labels.tolist()])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro-F1: {f1:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_sorted = list(range(len(class_to_idx)))
    display_labels = [id2class[i] for i in labels_sorted]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels_sorted,
        display_labels=display_labels,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )
    disp.figure_.tight_layout()
    fig_path = out_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved:", fig_path)

    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": args.data_dir,
        "weights": args.weights,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "num_classes": int(len(class_to_idx)),
        "classes": [id2class[i] for i in labels_sorted],
        "confusion_matrix_png": fig_path.as_posix(),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
