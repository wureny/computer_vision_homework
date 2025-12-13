# Data folder

This project expects an **ImageFolder** style dataset.

## Option A: raw folder (auto split)

Put images under:

```
data/raw/<class_name>/*
```

Then:

```bash
python prepare_dataset.py --raw_dir data/raw --out_dir data --train 0.8 --val 0.1 --test 0.1
```

## Option B: manual split

Provide:

```
data/train/<class_name>/*
data/val/<class_name>/*
data/test/<class_name>/*
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

