from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


METRICS_CSV = "experiment_metrics.csv"
FINAL_METRICS = "final_metrics.txt"
PSNR_LOG = "psnr.txt"


@dataclass(slots=True)
class Experiment:
    name: str
    path: Path
    rows: list[dict[str, str]]


def find_experiments(inputs: Iterable[Path]) -> list[Experiment]:
    experiments: list[Experiment] = []
    seen: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        candidates: list[Path] = []
        if (path / METRICS_CSV).exists() or (path / FINAL_METRICS).exists() or (path / PSNR_LOG).exists():
            candidates.append(path)
        elif path.exists():
            candidates.extend(sorted({item.parent for item in path.rglob(METRICS_CSV)}))
            candidates.extend(sorted({item.parent for item in path.rglob(FINAL_METRICS)}))
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            rows = load_metric_rows(candidate)
            if rows:
                experiments.append(Experiment(candidate.name, candidate, rows))
    return experiments


def load_metric_rows(exp_dir: Path) -> list[dict[str, str]]:
    csv_path = exp_dir / METRICS_CSV
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows = parse_final_metrics(exp_dir / FINAL_METRICS)
    rows.extend(parse_psnr_log(exp_dir / PSNR_LOG))
    return rows


def parse_final_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"FINAL ITERATION\s+(?P<iteration>\d+)\s+-\s+(?P<split>\S+).*?"
        r"PSNR:\s*(?P<psnr>[^\n]+).*?"
        r"SSIM:\s*(?P<ssim>[^\n]+).*?"
        r"LPIPS:\s*(?P<lpips>[^\n]+).*?"
        r"NUM_GAUSSIAN:\s*(?P<num_gaussians>\d+)\s*(?:\nFPS:\s*(?P<fps>[^\n]+))?.*?"
        r"TRAIN_SECONDS:\s*(?P<wall_seconds>[-+0-9.eE]+|unknown)",
        re.DOTALL,
    )
    return [match.groupdict() for match in pattern.finditer(text)]


def parse_psnr_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    points_by_iter: dict[str, str] = {}
    pending: dict[tuple[str, str], dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        count_match = re.search(r"\[ITER\s+(\d+)\]\s+NUM GAUSSIAN:\s+(\d+)", line)
        if count_match:
            points_by_iter[count_match.group(1)] = count_match.group(2)
            continue
        metric_match = re.search(r"\[ITER\s+(\d+)\]\s+Evaluating\s+(\S+):\s+L1\s+(\S+)\s+PSNR\s+(\S+)", line)
        if metric_match:
            iteration, split, l1_value, psnr_value = metric_match.groups()
            pending[(iteration, split)] = {
                "iteration": iteration,
                "split": split,
                "loss_l1": l1_value,
                "psnr": psnr_value,
                "num_gaussians": points_by_iter.get(iteration, ""),
            }
            continue
        perceptual_match = re.search(r"\[ITER\s+(\d+)\]\s+Evaluating\s+(\S+):\s+SSIM\s+(\S+)\s+LPIPS\s+(\S+)", line)
        if perceptual_match:
            iteration, split, ssim_value, lpips_value = perceptual_match.groups()
            row = pending.pop((iteration, split), {"iteration": iteration, "split": split})
            row.update({"ssim": ssim_value, "lpips": lpips_value})
            rows.append(row)
    rows.extend(pending.values())
    return rows


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def metric_points(exp: Experiment, metric: str, split_prefixes: tuple[str, ...] | None = None) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for row in exp.rows:
        split = str(row.get("split") or "")
        if split_prefixes and not any(split.startswith(prefix) for prefix in split_prefixes):
            continue
        iteration = to_float(row.get("iteration"))
        value = to_float(row.get(metric))
        if iteration is None or value is None:
            continue
        series.setdefault(split or "value", []).append((iteration, value))
    for values in series.values():
        values.sort()
    return series


def plot_curve(experiments: list[Experiment], metric: str, output: Path, title: str, ylabel: str, split_prefixes: tuple[str, ...] | None = None) -> bool:
    plotted = False
    plt.figure(figsize=(7.0, 4.2))
    for exp in experiments:
        series = metric_points(exp, metric, split_prefixes)
        for split, points in series.items():
            if not points:
                continue
            x_values, y_values = zip(*points)
            label = exp.name if len(experiments) == 1 and len(series) == 1 else f"{exp.name}:{split}"
            plt.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3, label=label)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def final_summary(exp: Experiment) -> dict[str, str]:
    summary = {"experiment": exp.name, "path": str(exp.path)}
    rows = [row for row in exp.rows if str(row.get("split") or "").startswith("test")]
    if not rows:
        rows = exp.rows
    rows = sorted(rows, key=lambda row: to_float(row.get("iteration")) or -1)
    if rows:
        final = rows[-1]
        for key in ("iteration", "psnr", "ssim", "lpips", "num_gaussians", "fps", "wall_seconds"):
            if final.get(key) not in (None, ""):
                value = to_float(final[key])
                summary[key] = str(value if value is not None else final[key])
    if not summary.get("wall_seconds"):
        train_rows = [row for row in exp.rows if row.get("split") == "train" and to_float(row.get("wall_seconds")) is not None]
        if train_rows:
            summary["wall_seconds"] = str(max(to_float(row.get("wall_seconds")) or 0.0 for row in train_rows))
    if not summary.get("fps"):
        train_rows = [row for row in exp.rows if row.get("split") == "train" and to_float(row.get("fps")) is not None]
        if train_rows:
            final_train_row = sorted(train_rows, key=lambda row: to_float(row.get("iteration")) or -1)[-1]
            summary["fps"] = str(to_float(final_train_row.get("fps")))
    return summary


def plot_training_time(experiments: list[Experiment], output: Path) -> bool:
    summaries = [final_summary(exp) for exp in experiments]
    points = [(item["experiment"], to_float(item.get("wall_seconds"))) for item in summaries]
    points = [(name, value) for name, value in points if value is not None]
    if not points:
        return False
    names, values = zip(*points)
    plt.figure(figsize=(max(6.0, len(names) * 1.2), 4.2))
    plt.bar(names, values, color="#4c78a8")
    plt.ylabel("Seconds")
    plt.title("Training Time")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def load_blur_series(exp: Experiment) -> list[tuple[int, float]]:
    candidates = [
        exp.path / "blur_labels.json",
        exp.path.parent / "blur_labels.json",
        exp.path / "metrics.json",
        exp.path.parent / "metrics.json",
        exp.path.parent.parent / "metrics.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        labels = payload.get("labels") or payload.get("blur_frame_registry")
        points = blur_points_from_labels(labels)
        if points:
            return points
    return []


def blur_points_from_labels(labels: object) -> list[tuple[int, float]]:
    if not isinstance(labels, dict):
        return []
    points: list[tuple[int, float]] = []
    type_strength = {"sharp": 0.0, "none": 0.0, "motion": 1.0, "defocus": 1.0, "mixed": 1.0}
    for index, key in enumerate(sorted(labels)):
        item = labels[key]
        value = None
        if isinstance(item, dict):
            for metric in ("blur_weight", "blurry_patch_ratio", "raw_score", "sharp_score", "laplacian"):
                value = to_float(item.get(metric))
                if value is not None:
                    break
            if value is None:
                value = type_strength.get(str(item.get("blur_type") or item.get("kind") or "").lower())
        else:
            value = type_strength.get(str(item).lower())
        if value is not None:
            points.append((index, value))
    return points


def plot_blur_intensity(experiments: list[Experiment], output: Path) -> bool:
    plotted = False
    plt.figure(figsize=(7.0, 4.2))
    for exp in experiments:
        points = load_blur_series(exp)
        if not points:
            continue
        x_values, y_values = zip(*points)
        plt.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=3, label=exp.name)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title("Blur Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Detector score or type strength")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def plot_ablation(experiments: list[Experiment], output: Path, metric: str) -> bool:
    points = []
    for exp in experiments:
        value = to_float(final_summary(exp).get(metric))
        if value is not None:
            points.append((exp.name, value))
    if not points:
        return False
    names, values = zip(*points)
    plt.figure(figsize=(max(6.0, len(names) * 1.2), 4.2))
    plt.bar(names, values, color="#59a14f")
    plt.ylabel(metric.upper())
    plt.title(f"Ablation by {metric.upper()}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def write_summary_tables(experiments: list[Experiment], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [final_summary(exp) for exp in experiments]
    fields = ["experiment", "iteration", "psnr", "ssim", "lpips", "num_gaussians", "fps", "wall_seconds", "path"]
    csv_path = output_dir / "experiment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})
    md_path = output_dir / "experiment_summary.md"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for summary in summaries:
        lines.append("| " + " | ".join(str(summary.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def generate_plots(experiments: list[Experiment], output_dir: Path, *, fmt: str = "png", ablation_metric: str = "psnr") -> list[Path]:
    if plt is None:
        raise RuntimeError(f"matplotlib is required to generate figures: {MATPLOTLIB_IMPORT_ERROR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    specs = [
        ("psnr", "PSNR", "PSNR", ("test", "train_sharp", "train")),
        ("ssim", "SSIM", "SSIM", ("test", "train_sharp")),
        ("lpips", "LPIPS", "LPIPS", ("test", "train_sharp")),
        ("num_gaussians", "Gaussian Count", "Gaussians", ("train",)),
        ("loss_total", "Training Loss", "Loss", ("train",)),
        ("vram_reserved_mb", "VRAM Usage", "MB", ("train",)),
        ("fps", "Training FPS", "FPS", ("train",)),
    ]
    for metric, title, ylabel, splits in specs:
        path = output_dir / f"{metric}_curve.{fmt}"
        if plot_curve(experiments, metric, path, title, ylabel, splits):
            generated.append(path)
    time_path = output_dir / f"training_time_bar.{fmt}"
    if plot_training_time(experiments, time_path):
        generated.append(time_path)
    blur_path = output_dir / f"blur_intensity_line.{fmt}"
    if plot_blur_intensity(experiments, blur_path):
        generated.append(blur_path)
    ablation_path = output_dir / f"ablation_{ablation_metric}_bar.{fmt}"
    if plot_ablation(experiments, ablation_path, ablation_metric):
        generated.append(ablation_path)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot DashDeblurGroupGS experiment curves for thesis figures.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Experiment output dirs or parent dirs containing experiment_metrics.csv.")
    parser.add_argument("-o", "--output", type=Path, default=Path("thesis_assets") / "figures")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--ablation-metric", default="psnr", choices=["psnr", "ssim", "lpips", "num_gaussians"])
    args = parser.parse_args()

    experiments = find_experiments(args.inputs)
    if not experiments:
        raise SystemExit("No experiment metrics found. Pass a model output directory or a parent directory.")
    try:
        generated = generate_plots(experiments, args.output, fmt=args.format, ablation_metric=args.ablation_metric)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    write_summary_tables(experiments, args.output.parent / "tables")
    print(f"Loaded {len(experiments)} experiment(s)")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
