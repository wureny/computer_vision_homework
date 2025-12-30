# Meme Template Classification (Final Project - Computer Vision)

This repo implements a **simple image classification** project: classify a meme image into one of several meme template categories (e.g., Pepe/Wojak/Doge/...).

The implementation targets a **normal undergraduate level**: transfer learning with **MobileNetV2 (ImageNet pretrained)** + standard training/evaluation/visualization.

## 1) Requirements: software

- Python >= 3.9
- PyTorch + TorchVision

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Pretrained models

- Put the trained weights here: `pretrained/mobilenetv2_meme_best.pt`
- The training script will generate it automatically (see below).

## 3) Preparation for testing

### Dataset format

This repo supports two dataset modes:

- **Full dataset (not committed)**: put your collected images under `data/raw/` and run `prepare_dataset.py`.
- **Demo dataset (committed by you)**: put a tiny teacher-friendly set under `data/demo/` so the TA can run `eval.py` directly after cloning (no private dataset needed).

Option A (recommended): prepare from a raw folder and auto-split:

```
data/raw/
  Pepe/xxx.jpg
  Wojak/yyy.png
  Doge/zzz.jpeg
  ...
```

Then run:

```bash
python prepare_dataset.py --raw_dir data/raw --out_dir data --train 0.8 --val 0.1 --test 0.1
```

Option B: manually provide:

```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg
data/test/<class_name>/*.jpg
```

### Teacher/TA quick demo (recommended to include in GitHub)

1) Prepare a small demo dataset:

- Put a few images per class under `data/demo/raw/<class_name>/*` (e.g. 3--10 images per class).
- Split it:

```bash
python3 prepare_dataset.py --raw_dir data/demo/raw --out_dir data/demo --train 0.7 --val 0.15 --test 0.15
```

2) Train a demo checkpoint and copy to `pretrained/`:

```bash
python3 train.py --data_dir data/demo --epochs 10 --batch_size 16 --lr 1e-4 --freeze_backbone
```

3) Evaluate (also saves `metrics.json` + confusion matrix figure under `outputs/`):

```bash
python3 eval.py --data_dir data/demo --weights pretrained/mobilenetv2_meme_best.pt --out_dir outputs/eval_demo
```

### Train (also produces pretrained weights)

```bash
python train.py --data_dir data --epochs 15 --batch_size 32 --lr 1e-4
```

Outputs (per run) are saved under `outputs/`.

### Test / Evaluate

```bash
python eval.py --data_dir data --weights pretrained/mobilenetv2_meme_best.pt
```

### Predict a single image

```bash
python predict.py --weights pretrained/mobilenetv2_meme_best.pt --image path/to/image.jpg
```

## Notes

- The final report (English) should include this GitHub URL in the abstract (per assignment requirements).
- This repo does **not** download datasets automatically (network may be restricted). You should place images under `data/raw/` (or `data/train|val|test/`).
