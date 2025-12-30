# Demo dataset (for TA quick run)

The assignment asks for a GitHub repo that can be tested easily. A full meme dataset is usually too large or may have licensing concerns, so this project supports a tiny **demo dataset** that you can commit.

## Folder layout

Put a few images per class under:

```
data/demo/raw/<class_name>/*
```

Example:

```
data/demo/raw/
  Pepe/...
  Doge/...
  Wojak/...
```

Then split into train/val/test:

```bash
python3 prepare_dataset.py --raw_dir data/demo/raw --out_dir data/demo --train 0.7 --val 0.15 --test 0.15
```

After splitting, the structure becomes:

```
data/demo/train/<class_name>/*
data/demo/val/<class_name>/*
data/demo/test/<class_name>/*
```

## Recommended size

- Minimum to run: 3 images per class.
- Recommended: 5--15 images per class (still small, but produces non-trivial metrics).

