#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "data/demo/raw" ]]; then
  echo "Missing demo raw images folder: data/demo/raw"
  echo "Put images under: data/demo/raw/<class_name>/*"
  exit 1
fi

mkdir -p .torch_cache .mplconfig .cache/fontconfig
export TORCH_HOME="$ROOT_DIR/.torch_cache"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"

python3 prepare_dataset.py --raw_dir data/demo/raw --out_dir data/demo --train 0.7 --val 0.15 --test 0.15
python3 train.py --data_dir data/demo --epochs 10 --batch_size 16 --lr 1e-4 --freeze_backbone
python3 eval.py --data_dir data/demo --weights pretrained/mobilenetv2_meme_best.pt --out_dir outputs/eval_demo --num_workers 0
python3 report/update_results.py --metrics outputs/eval_demo/metrics.json --out report/results.tex --assets_dir report/figures
python3 report/update_dataset.py --splits data/demo/splits.json --out report/dataset.tex

echo "Done."
echo "- Weights: pretrained/mobilenetv2_meme_best.pt"
echo "- Metrics: outputs/eval_demo/metrics.json"
echo "- Confusion matrix: outputs/eval_demo/confusion_matrix.png"
echo "- Report macros: report/results.tex"
echo "- Dataset table: report/dataset.tex"
