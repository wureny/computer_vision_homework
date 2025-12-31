"""
Evaluation entry for a trained meme template classifier checkpoint.

Key steps:
- Load checkpoint (model + class_to_idx)
- Load ImageFolder dataset from `data_dir/test`
- Run inference to compute accuracy and macro-F1
- Save confusion matrix and a machine-readable `metrics.json`
- (Optional) export misclassified examples for qualitative analysis
"""

import argparse
import json
import time
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_utils import idx_to_class, load_checkpoint


def make_transforms() -> transforms.Compose:
    # Match validation/test preprocessing used during training.
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    parser.add_argument(
        "--errors_dir",
        type=str,
        default="",
        help="If set, copy misclassified images and write errors.json for qualitative analysis.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_checkpoint(args.weights, device=device)
    model = bundle.model
    class_to_idx = bundle.class_to_idx
    id2class = idx_to_class(class_to_idx)

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        raise SystemExit(f"Expected folder not found: {test_dir}")

    class ImageFolderWithPath(datasets.ImageFolder):
        # Keep the original file path so we can export error cases later.
        def __getitem__(self, index: int):
            image, label = super().__getitem__(index)
            path, _ = self.samples[index]
            return image, label, path

    test_ds = ImageFolderWithPath(test_dir.as_posix(), transform=make_transforms())
    pin_memory = device == "cuda"
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    # Map indices used by ImageFolder(test) to indices used by the trained checkpoint.
    # This makes evaluation robust even if folder order differs between train/test.
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

    y_prob: list[list[float]] = []
    paths: list[str] = []

    for images, labels, batch_paths in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend([test_idx_to_ckpt_idx[int(i)] for i in labels.tolist()])
        y_prob.extend(probs.tolist())
        paths.extend(list(batch_paths))

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

    if args.errors_dir:
        # Save misclassified images so the report can show failure cases.
        errors_dir = Path(args.errors_dir)
        errors_dir.mkdir(parents=True, exist_ok=True)
        errors: list[dict[str, object]] = []
        for true_i, pred_i, prob_vec, p in zip(y_true, y_pred, y_prob, paths, strict=True):
            if int(true_i) == int(pred_i):
                continue
            src = Path(p)
            dst = errors_dir / f"true_{id2class[int(true_i)]}__pred_{id2class[int(pred_i)]}__{src.name}"
            try:
                shutil.copy2(src, dst)
            except Exception:
                # If copy fails (e.g., permission), still record the path.
                dst = src
            errors.append(
                {
                    "path": str(src),
                    "copied_to": str(dst),
                    "true": id2class[int(true_i)],
                    "pred": id2class[int(pred_i)],
                    "pred_confidence": float(max(prob_vec)),
                }
            )
        (errors_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics["num_errors"] = int(len(errors))
        metrics["errors_dir"] = errors_dir.as_posix()

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
