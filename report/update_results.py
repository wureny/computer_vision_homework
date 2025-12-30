import argparse
import json
import shutil
from pathlib import Path


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report/results.tex from eval metrics.json")
    parser.add_argument("--metrics", type=str, required=True, help="e.g. outputs/eval_demo/metrics.json")
    parser.add_argument("--out", type=str, default="report/results.tex")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="report/figures",
        help="If set, copy confusion_matrix.png into this folder and reference it in LaTeX.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    acc = float(data.get("accuracy", 0.0)) * 100.0
    f1 = float(data.get("macro_f1", 0.0)) * 100.0
    classes = data.get("classes", [])
    cm_png = str(data.get("confusion_matrix_png", ""))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_tex_path = cm_png
    if cm_png and args.assets_dir:
        src = Path(cm_png)
        if src.exists() and src.is_file():
            assets_dir = Path(args.assets_dir)
            assets_dir.mkdir(parents=True, exist_ok=True)
            dst = assets_dir / "confusion_matrix.png"
            shutil.copy2(src, dst)
            try:
                cm_tex_path = dst.relative_to(out_path.parent).as_posix()
            except ValueError:
                cm_tex_path = dst.as_posix()

    lines = [
        "% Auto-generated. Do not edit by hand.",
        f"\\newcommand{{\\ResultAccuracy}}{{{acc:.2f}\\%}}",
        f"\\newcommand{{\\ResultMacroFone}}{{{f1:.2f}\\%}}",
        f"\\newcommand{{\\ResultNumClasses}}{{{len(classes)}}}",
        f"\\newcommand{{\\ResultClasses}}{{{tex_escape(', '.join(map(str, classes)))}}}",
        f"\\newcommand{{\\ResultConfusionMatrix}}{{{tex_escape(cm_tex_path)}}}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
