import argparse
import json
from pathlib import Path


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}\\%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report/ablation.tex from two metrics.json files")
    parser.add_argument("--freeze", type=str, required=True, help="metrics.json for freeze_backbone run")
    parser.add_argument("--finetune", type=str, required=True, help="metrics.json for full fine-tune run")
    parser.add_argument("--out", type=str, default="report/ablation.tex")
    args = parser.parse_args()

    freeze = read_metrics(Path(args.freeze))
    finetune = read_metrics(Path(args.finetune))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Auto-generated. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study on backbone fine-tuning.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Setting & Accuracy & Macro-F1 \\\\",
        "\\midrule",
        f"Freeze backbone & {fmt_pct(float(freeze.get('accuracy', 0.0)))} & {fmt_pct(float(freeze.get('macro_f1', 0.0)))} \\\\",
        f"Fine-tune backbone & {fmt_pct(float(finetune.get('accuracy', 0.0)))} & {fmt_pct(float(finetune.get('macro_f1', 0.0)))} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

