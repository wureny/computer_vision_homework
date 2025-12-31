#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

mkdir -p .torch_cache .mplconfig .cache/fontconfig
export TORCH_HOME="$ROOT_DIR/.torch_cache"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"

if [[ ! -d "data/raw" ]]; then
  echo "Missing data/raw. Put your full dataset under data/raw/<class_name>/*"
  exit 1
fi

# 1) Split full dataset (more test samples for more convincing metrics)
$PYTHON_BIN prepare_dataset.py --raw_dir data/raw --out_dir data --train 0.6 --val 0.2 --test 0.2 --seed 42 --clean

# 2) Ablation: freeze vs finetune
$PYTHON_BIN train.py --data_dir data --epochs 25 --batch_size 16 --lr 1e-4 --weight_decay 1e-4 --freeze_backbone --num_workers 0 --weights_out pretrained/mobilenetv2_meme_freeze.pt
$PYTHON_BIN eval.py --data_dir data --weights pretrained/mobilenetv2_meme_freeze.pt --out_dir outputs/eval_freeze --num_workers 0 --errors_dir outputs/eval_freeze/errors

$PYTHON_BIN train.py --data_dir data --epochs 25 --batch_size 16 --lr 1e-4 --weight_decay 1e-4 --num_workers 0 --weights_out pretrained/mobilenetv2_meme_best.pt
$PYTHON_BIN eval.py --data_dir data --weights pretrained/mobilenetv2_meme_best.pt --out_dir outputs/eval_finetune --num_workers 0 --errors_dir outputs/eval_finetune/errors

# 3) Update report auto-filled tables/figures
$PYTHON_BIN report/update_results.py --metrics outputs/eval_finetune/metrics.json --out report/results.tex --assets_dir report/figures
$PYTHON_BIN report/update_dataset.py --splits data/splits.json --out report/dataset.tex
$PYTHON_BIN report/update_ablation.py --freeze outputs/eval_freeze/metrics.json --finetune outputs/eval_finetune/metrics.json --out report/ablation.tex

echo "Done."
echo "- Best weights: pretrained/mobilenetv2_meme_best.pt"
echo "- Freeze weights (ablation): pretrained/mobilenetv2_meme_freeze.pt"
echo "- Report: report/final_report.tex (compile in Overleaf)"
