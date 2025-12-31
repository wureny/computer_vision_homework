# Meme Template Classification (Final Project - Computer Vision)

This repo implements a **simple image classification** project: classify a meme image into one of several meme template categories (e.g., Pepe/Wojak/Doge/...).

The implementation targets a **normal undergraduate level**: transfer learning with **MobileNetV2 (ImageNet pretrained)** + standard training/evaluation/visualization.

## 1) Requirements: software

- Python >= 3.9
- PyTorch + TorchVision

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- For macOS, `torch` works well with CPU/MPS; this project sets some cache env vars in scripts to avoid permission issues.

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
python3 prepare_dataset.py --raw_dir data/raw --out_dir data --train 0.8 --val 0.1 --test 0.1
```

Option B: manually provide:

```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg
data/test/<class_name>/*.jpg
```

### Teacher/TA quick demo (recommended to include in GitHub)

The repo already includes a tiny demo dataset under `data/demo/` and a trained weight under `pretrained/`.

After cloning, the TA can run:

```bash
bash scripts/quick_demo.sh
```

This evaluates `pretrained/mobilenetv2_meme_best.pt` on `data/demo/test` and saves:

- `outputs/eval_demo/metrics.json`
- `outputs/eval_demo/confusion_matrix.png`

If you want to regenerate the demo split and retrain on it:

```bash
bash scripts/train_demo.sh
```

### Train (also produces pretrained weights)

```bash
python3 train.py --data_dir data --epochs 15 --batch_size 32 --lr 1e-4
```

Outputs (per run) are saved under `outputs/`.

### Full experiments (for report reproduction)

If you have a larger dataset under `data/raw/`, this script runs a more convincing split and an ablation study (freeze vs fine-tune), then updates the auto-filled LaTeX tables:

```bash
bash scripts/run_full_experiments.sh
```

### Test / Evaluate

```bash
python3 eval.py --data_dir data --weights pretrained/mobilenetv2_meme_best.pt
```

### Predict a single image

```bash
python3 predict.py --weights pretrained/mobilenetv2_meme_best.pt --image path/to/image.jpg
```

## Notes

- The final report (English) should include this GitHub URL in the abstract (per assignment requirements).
- This repo does **not** download datasets automatically (network may be restricted). You should place images under `data/raw/` (or `data/train|val|test/`).
