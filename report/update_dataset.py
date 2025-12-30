import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report/dataset.tex from data splits.json")
    parser.add_argument("--splits", type=str, required=True, help="e.g. data/demo/splits.json")
    parser.add_argument("--out", type=str, default="report/dataset.tex")
    args = parser.parse_args()

    splits_path = Path(args.splits)
    stats = json.loads(splits_path.read_text(encoding="utf-8"))

    rows = []
    total_train = total_val = total_test = 0
    for cls in sorted(stats.keys()):
        r = stats[cls]
        tr, va, te = int(r.get("train", 0)), int(r.get("val", 0)), int(r.get("test", 0))
        total_train += tr
        total_val += va
        total_test += te
        rows.append((cls, tr, va, te, tr + va + te))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Auto-generated. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dataset split statistics (number of images per class).}",
        "\\label{tab:dataset}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Class & Train & Val & Test & Total \\\\",
        "\\midrule",
    ]
    for cls, tr, va, te, tot in rows:
        safe_cls = str(cls).replace("_", "\\_")
        lines.append(f"{safe_cls} & {tr} & {va} & {te} & {tot} \\\\")
    lines += [
        "\\midrule",
        f"Total & {total_train} & {total_val} & {total_test} & {total_train + total_val + total_test} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

