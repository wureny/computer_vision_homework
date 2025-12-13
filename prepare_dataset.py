import argparse
import json
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(class_dir: Path) -> list[Path]:
    images: list[Path] = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def copy_files(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split data/raw into train/val/test folders.")
    parser.add_argument("--raw_dir", type=str, required=True, help="e.g. data/raw")
    parser.add_argument("--out_dir", type=str, required=True, help="e.g. data")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove existing train/val/test before splitting.")
    args = parser.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("--train + --val + --test must sum to 1.0")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw_dir not found: {raw_dir}")

    for split in ["train", "val", "test"]:
        split_dir = out_dir / split
        if args.clean and split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    class_names = [p.name for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    class_names = sorted(class_names)
    if not class_names:
        raise SystemExit(f"No class subfolders found under: {raw_dir}")

    random.seed(args.seed)
    stats: dict[str, dict[str, int]] = {}

    for class_name in class_names:
        class_dir = raw_dir / class_name
        images = iter_images(class_dir)
        if not images:
            continue
        random.shuffle(images)

        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train : n_train + n_val]
        test_files = images[n_train + n_val :]
        assert len(test_files) == n_test

        copy_files(train_files, out_dir / "train" / class_name)
        copy_files(val_files, out_dir / "val" / class_name)
        copy_files(test_files, out_dir / "test" / class_name)

        stats[class_name] = {"train": len(train_files), "val": len(val_files), "test": len(test_files)}

    (out_dir / "splits.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done. Split summary saved to:", out_dir / "splits.json")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

