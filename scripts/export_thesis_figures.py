from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_experiments


LOG_NAMES = (
    "experiment_metrics.csv",
    "psnr.txt",
    "final_metrics.txt",
    "cfg_args",
    "train_config.txt",
    "metrics.json",
    "blur_labels.json",
    "task.log",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_logs(exp: plot_experiments.Experiment, logs_dir: Path) -> int:
    copied = 0
    candidates = [exp.path, exp.path.parent, exp.path.parent.parent]
    for base in candidates:
        for name in LOG_NAMES:
            if copy_if_exists(base / name, logs_dir / exp.name / name):
                copied += 1
    return copied


def visual_candidates(exp: plot_experiments.Experiment) -> list[Path]:
    roots = [
        exp.path / "TEST",
        exp.path / "TRAIN",
        exp.path / "test",
        exp.path / "train",
    ]
    roots.extend(path for path in exp.path.iterdir() if path.is_dir() and path.name.upper().startswith(("TEST", "TRAIN")))
    images: list[Path] = []
    for root in sorted(set(roots)):
        if root.exists():
            images.extend(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(images, key=lambda path: (path.parent.name, path.name))


def copy_visuals(exp: plot_experiments.Experiment, visual_dir: Path, max_images: int) -> int:
    copied = 0
    for source in visual_candidates(exp)[:max_images]:
        relative = source.relative_to(exp.path)
        if copy_if_exists(source, visual_dir / exp.name / relative):
            copied += 1
    return copied


def export_assets(inputs: list[Path], assets_dir: Path, *, fmt: str, ablation_metric: str, max_visuals: int) -> dict[str, int]:
    experiments = plot_experiments.find_experiments(inputs)
    if not experiments:
        raise SystemExit("No experiment metrics found. Pass a model output directory or a parent directory.")

    figures_dir = assets_dir / "figures"
    tables_dir = assets_dir / "tables"
    visual_dir = assets_dir / "visual_comparisons"
    logs_dir = assets_dir / "logs"
    for directory in (figures_dir, tables_dir, visual_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        figures = plot_experiments.generate_plots(experiments, figures_dir, fmt=fmt, ablation_metric=ablation_metric)
    except RuntimeError as exc:
        print(exc)
        figures = []
    tables = plot_experiments.write_summary_tables(experiments, tables_dir)
    log_count = sum(copy_logs(exp, logs_dir) for exp in experiments)
    visual_count = sum(copy_visuals(exp, visual_dir, max_visuals) for exp in experiments)

    return {
        "experiments": len(experiments),
        "figures": len(figures),
        "tables": len(tables),
        "logs": log_count,
        "visuals": visual_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect experiment outputs into thesis_assets.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Experiment output dirs or parent dirs containing experiment_metrics.csv.")
    parser.add_argument("-o", "--output", type=Path, default=Path("thesis_assets"))
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--ablation-metric", default="psnr", choices=["psnr", "ssim", "lpips", "num_gaussians"])
    parser.add_argument("--max-visuals", type=int, default=24)
    args = parser.parse_args()

    counts = export_assets(
        args.inputs,
        args.output,
        fmt=args.format,
        ablation_metric=args.ablation_metric,
        max_visuals=max(0, args.max_visuals),
    )
    print(f"Exported {counts['experiments']} experiment(s) to {args.output}")
    print(f"figures={counts['figures']} tables={counts['tables']} logs={counts['logs']} visuals={counts['visuals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
