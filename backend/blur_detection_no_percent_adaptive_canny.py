import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

BLUR_EFFECT_BLURRY_THRESHOLD = 0.35
BLUR_EFFECT_SHARP_THRESHOLD = 0.22


def resolve_worker_count(workers=None, image_count=0):
    if workers is not None:
        workers = int(workers)
        if workers > 0:
            return workers

    cpu_count = os.cpu_count() or 1
    if image_count <= 1:
        return 1
    return max(1, min(8, cpu_count, image_count))


def read_gray_image(path, max_size=1600):
    """
    Windows 中文路径兼容读取。
    不使用 cv2.imread(str(path))，避免中文路径乱码。
    """
    path = Path(path)

    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        raise ValueError(f"Cannot read image bytes: {path}")

    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot decode image: {path}")

    h, w = img.shape[:2]
    scale = max(h, w) / max_size

    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def calc_blur_effect_from_gray(gray, h_size=11):
    from skimage import measure

    gray_f = gray.astype(np.float32)
    max_value = float(np.max(gray_f)) if gray_f.size else 0.0
    if max_value > 1.0:
        gray_f = gray_f / 255.0

    score = measure.blur_effect(gray_f, h_size=h_size)
    return float(score)


def classify_blur_effect(score):
    if is_missing(score):
        return "uncertain"
    if score >= BLUR_EFFECT_BLURRY_THRESHOLD:
        return "blurry"
    if score <= BLUR_EFFECT_SHARP_THRESHOLD:
        return "sharp"
    return "uncertain"


def estimate_noise(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(1.4826 * mad)


def patch_sharpness_scores(
    gray,
    patch_size=64,
    stride=64,
    min_texture_ratio=0.03,
    edge_thresh=18,
):
    h, w = gray.shape

    if h < patch_size or w < patch_size:
        patch_size = min(h, w)
        stride = patch_size

    lap_scores = []
    ten_scores = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = gray[y:y + patch_size, x:x + patch_size]

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)

            texture_ratio = float(np.mean(grad > edge_thresh))

            if texture_ratio < min_texture_ratio:
                continue

            lap = cv2.Laplacian(patch, cv2.CV_32F)
            lap_var = float(lap.var())

            tenengrad = float(np.mean(gx * gx + gy * gy))

            lap_scores.append(lap_var)
            ten_scores.append(tenengrad)

    if len(lap_scores) == 0:
        return None

    return (
        np.array(lap_scores, dtype=np.float32),
        np.array(ten_scores, dtype=np.float32),
    )


def directionality_score(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ex = float(np.mean(gx * gx))
    ey = float(np.mean(gy * gy))

    small = max(min(ex, ey), 1e-6)
    large = max(ex, ey)

    return large / small


def fft_high_freq_ratio(gray):
    """
    原始幅值高频比例，保留作为辅助。
    """
    img = gray.astype(np.float32) / 255.0
    h, w = img.shape

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = np.sqrt(cx ** 2 + cy ** 2)
    high_mask = r > max_r * 0.35

    total_energy = float(np.sum(mag) + 1e-8)
    high_energy = float(np.sum(mag[high_mask]))

    return high_energy / total_energy


def fft_power_high_freq_ratio(gray):
    """
    功率谱高频比例。
    对整体模糊更敏感。
    越低越模糊。
    """
    img = gray.astype(np.float32) / 255.0
    h, w = img.shape

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = np.sqrt(cx ** 2 + cy ** 2)
    high_mask = r > max_r * 0.35

    total_power = float(np.sum(power) + 1e-8)
    high_power = float(np.sum(power[high_mask]))

    return high_power / total_power


def adaptive_canny_edges(gray, grad=None, low_percentile=70, high_percentile=90):
    """
    自适应 Canny 边缘检测。
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if grad is None:
        g = blurred.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

    valid_grad = grad[np.isfinite(grad) & (grad > 0)]

    if valid_grad.size >= 32:
        lower = float(np.percentile(valid_grad, low_percentile))
        upper = float(np.percentile(valid_grad, high_percentile))

        if upper > lower and upper > 1e-6:
            return cv2.Canny(blurred, lower, upper)

    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    if upper <= lower:
        upper = min(255, lower + 1)

    return cv2.Canny(blurred, lower, upper)


def edge_spread_width(gray, max_edges=8000):
    """
    边缘扩散宽度。

    清晰图：边缘过渡窄，edge_width 小。
    模糊图：边缘被拉宽，edge_width 大。
    """
    g = gray.astype(np.float32)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    if grad.size == 0:
        return np.nan, np.nan, 0

    high_grad = np.percentile(grad, 80)
    if high_grad <= 1e-6:
        return np.nan, np.nan, 0

    edges = adaptive_canny_edges(gray, grad=grad)

    ys, xs = np.where((edges > 0) & (grad > high_grad))

    if len(xs) == 0:
        return np.nan, np.nan, 0

    if len(xs) > max_edges:
        idx = np.linspace(0, len(xs) - 1, max_edges).astype(np.int32)
        xs = xs[idx]
        ys = ys[idx]

    h, w = gray.shape
    widths = []

    for y, x in zip(ys, xs):
        abs_gx = abs(float(gx[y, x]))
        abs_gy = abs(float(gy[y, x]))

        if abs_gx >= abs_gy:
            a = max(0, x - 20)
            b = min(w, x + 21)
            line = g[y, a:b]
        else:
            a = max(0, y - 20)
            b = min(h, y + 21)
            line = g[a:b, x]

        if len(line) < 8:
            continue

        diff = np.abs(np.diff(line))
        if diff.size == 0:
            continue

        peak = float(np.max(diff))
        if peak < 3:
            continue

        th = max(peak * 0.12, 2.0)
        support = np.where(diff > th)[0]

        if support.size == 0:
            continue

        width = int(support[-1] - support[0] + 1)

        if 1 <= width <= 40:
            widths.append(width)

    if len(widths) == 0:
        return np.nan, np.nan, 0

    return (
        float(np.median(widths)),
        float(np.percentile(widths, 75)),
        int(len(widths)),
    )


def safe_percentile(values, q):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.percentile(values, q))


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def nanmedian(values):
    valid = [value for value in values if not is_missing(value)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(np.asarray(valid, dtype=np.float64)))


def normalize_detected_blur_type(row):
    label = str(row.get("label") or "").strip().lower()
    blur_type = str(row.get("blur_type") or "").strip().lower()

    if label == "sharp" or blur_type in {"none", "sharp"}:
        return "sharp"
    if blur_type in {"defocus", "defocus_blur"}:
        return "defocus"
    if blur_type in {"motion", "motion_blur", "blur_unknown", "uncertain"}:
        return "motion"
    if label in {"blurry", "uncertain"}:
        return "motion"
    return "motion"


def apply_blur_effect_second_check_rows(rows):
    for row in rows:
        row["label_before_blur_effect"] = row.get("label")
        row["blur_type_before_blur_effect"] = row.get("blur_type")

        blur_scores = row.get("blur_effect")
        blur_effect_label = classify_blur_effect(blur_scores)
        row["blur_effect_label"] = blur_effect_label

        blur_type = str(row.get("blur_type") or "").strip().lower()
        if blur_type in {"defocus", "defocus_blur"}:
            row["label"] = "blurry"
            row["blur_type"] = "defocus_blur"
            row["blur_policy"] = "preserve_defocus"
            continue

        if blur_effect_label == "blurry":
            row["label"] = "blurry"
            row["blur_type"] = "motion_blur"
            row["blur_policy"] = "blur_effect_motion"
        else:
            row["label"] = "sharp"
            row["blur_type"] = "none"
            row["blur_policy"] = "blur_effect_clear"

    return rows


def apply_blur_effect_motion_policy_rows(rows):
    return apply_blur_effect_second_check_rows(rows)


def analyze_image(path):
    gray = read_gray_image(path)

    blur_effect = calc_blur_effect_from_gray(gray, h_size=11)
    blur_effect_label = classify_blur_effect(blur_effect)

    noise = estimate_noise(gray)
    directionality = directionality_score(gray)
    high_freq_ratio = fft_high_freq_ratio(gray)
    power_high_freq_ratio = fft_power_high_freq_ratio(gray)

    edge_width_median, edge_width_p75, edge_count = edge_spread_width(gray)

    patch_data = patch_sharpness_scores(gray)

    if patch_data is None:
        return {
            "path": str(path),
            "filename": path.name,
            "valid_patch_count": 0,
            "lap_scores_list": [],
            "lap_p10": np.nan,
            "lap_p20": np.nan,
            "lap_median": np.nan,
            "lap_p80": np.nan,
            "tenengrad_p20": np.nan,
            "tenengrad_median": np.nan,
            "directionality": directionality,
            "fft_high_ratio": high_freq_ratio,
            "fft_power_high_ratio": power_high_freq_ratio,
            "edge_width_median": edge_width_median,
            "edge_width_p75": edge_width_p75,
            "edge_count": edge_count,
            "noise": noise,
            "raw_score": np.nan,
            "blur_effect": blur_effect,
            "blur_effect_label": blur_effect_label,
        }

    lap_scores, ten_scores = patch_data

    lap_p10 = safe_percentile(lap_scores, 10)
    lap_p20 = safe_percentile(lap_scores, 20)
    lap_median = safe_percentile(lap_scores, 50)
    lap_p80 = safe_percentile(lap_scores, 80)

    ten_p20 = safe_percentile(ten_scores, 20)
    ten_median = safe_percentile(ten_scores, 50)

    raw_score = (
        0.75 * np.log1p(lap_p20)
        + 0.25 * np.log1p(ten_p20)
        + 8.0 * power_high_freq_ratio
        + 1.0 * high_freq_ratio
        - 0.015 * noise
    )

    if not np.isnan(edge_width_median):
        raw_score -= 0.06 * edge_width_median

    return {
        "path": str(path),
        "filename": path.name,
        "valid_patch_count": int(len(lap_scores)),
        "lap_scores_list": lap_scores.tolist(),
        "lap_p10": lap_p10,
        "lap_p20": lap_p20,
        "lap_median": lap_median,
        "lap_p80": lap_p80,
        "tenengrad_p20": ten_p20,
        "tenengrad_median": ten_median,
        "directionality": float(directionality),
        "fft_high_ratio": float(high_freq_ratio),
        "fft_power_high_ratio": float(power_high_freq_ratio),
        "edge_width_median": edge_width_median,
        "edge_width_p75": edge_width_p75,
        "edge_count": edge_count,
        "noise": float(noise),
        "raw_score": float(raw_score),
        "blur_effect": blur_effect,
        "blur_effect_label": blur_effect_label,
    }


def classify_result_rows(
    rows,
    motion_directionality_threshold=1.35,
    motion_soft_directionality_threshold=1.08,
    blurry_patch_percentile=35,
    blurry_patch_ratio_threshold=0.34,
    edge_width_absolute_threshold=10.0,
    motion_blur_weight_threshold=0.30,
    clear_force_weight_threshold=0.22,
    clear_force_lap_p20_threshold=120.0,
    clear_force_patch_ratio_threshold=0.12,
    possible_to_blurry_weight_threshold=0.28,
):
    rows = [dict(row) for row in rows]
    scores = [row.get("raw_score") for row in rows if not is_missing(row.get("raw_score"))]

    if len(scores) == 0:
        for row in rows:
            row["label"] = "uncertain"
            row["blur_type"] = "uncertain"
            row["blur_weight"] = 0.5
            row["blurry_patch_ratio"] = np.nan
        return apply_blur_effect_motion_policy_rows(rows)

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    median_fft = nanmedian([row.get("fft_high_ratio") for row in rows])
    median_power_fft = nanmedian([row.get("fft_power_high_ratio") for row in rows])
    median_lap = nanmedian([row.get("lap_p20") for row in rows])
    median_edge_width = nanmedian([row.get("edge_width_median") for row in rows])

    all_patch_laps = []
    for value in [row.get("lap_scores_list") for row in rows]:
        if isinstance(value, list):
            all_patch_laps.extend(value)

    patch_blur_th = float(np.percentile(all_patch_laps, blurry_patch_percentile)) if len(all_patch_laps) > 0 else median_lap

    for row in rows:
        s = row.get("raw_score", np.nan)
        if is_missing(s):
            row["label"] = "uncertain"
            row["blur_type"] = "uncertain"
            row["blur_weight"] = 0.5
            row["blurry_patch_ratio"] = np.nan
            continue

        blur_weight = (score_max - s) / max(score_max - score_min, 1e-6)
        blur_weight = float(np.clip(blur_weight, 0.0, 1.0))
        row["blur_weight"] = blur_weight

        directionality = row.get("directionality", np.nan)
        fft_high = row.get("fft_high_ratio", np.nan)
        power_fft = row.get("fft_power_high_ratio", np.nan)
        lap_p20 = row.get("lap_p20", np.nan)
        edge_width = row.get("edge_width_median", np.nan)
        patch_laps = row.get("lap_scores_list", [])

        if isinstance(patch_laps, list) and len(patch_laps) > 0 and not is_missing(patch_blur_th):
            blurry_patch_ratio = float(np.mean(np.array(patch_laps) < patch_blur_th))
        else:
            blurry_patch_ratio = np.nan
        row["blurry_patch_ratio"] = blurry_patch_ratio

        is_strong_directional = not is_missing(directionality) and directionality >= motion_directionality_threshold
        is_soft_directional = not is_missing(directionality) and directionality >= motion_soft_directionality_threshold
        low_freq_evidence = not is_missing(fft_high) and not is_missing(median_fft) and fft_high < median_fft
        low_power_freq_evidence = not is_missing(power_fft) and not is_missing(median_power_fft) and power_fft < median_power_fft
        low_lap_evidence = not is_missing(lap_p20) and not is_missing(median_lap) and lap_p20 < median_lap
        patch_blur_evidence = not is_missing(blurry_patch_ratio) and blurry_patch_ratio >= blurry_patch_ratio_threshold
        edge_width_absolute_evidence = not is_missing(edge_width) and edge_width >= edge_width_absolute_threshold
        edge_width_relative_evidence = (
            not is_missing(edge_width)
            and not is_missing(median_edge_width)
            and edge_width >= median_edge_width * 1.20
        )
        frequency_evidence = low_freq_evidence or low_power_freq_evidence

        clear_force_evidence = (
            blur_weight <= clear_force_weight_threshold
            and not is_missing(lap_p20)
            and lap_p20 >= clear_force_lap_p20_threshold
            and not is_missing(blurry_patch_ratio)
            and blurry_patch_ratio <= clear_force_patch_ratio_threshold
        )
        if clear_force_evidence:
            row["label"] = "sharp"
            row["blur_type"] = "none"
            continue

        strong_motion_blur_evidence = (
            is_strong_directional
            and (edge_width_absolute_evidence or (patch_blur_evidence and frequency_evidence))
        )
        soft_motion_blur_evidence = (
            is_soft_directional
            and edge_width_absolute_evidence
            and blur_weight >= motion_blur_weight_threshold
            and low_lap_evidence
        )
        defocus_blur_evidence = (
            patch_blur_evidence
            and low_lap_evidence
            and low_power_freq_evidence
            and (edge_width_absolute_evidence or edge_width_relative_evidence)
        )
        strong_blur_evidence = (
            edge_width_absolute_evidence
            and blur_weight >= possible_to_blurry_weight_threshold
            and (low_power_freq_evidence or patch_blur_evidence or low_lap_evidence)
        )
        possible_motion_evidence = (
            blur_weight >= possible_to_blurry_weight_threshold
            and is_soft_directional
            and edge_width_absolute_evidence
            and (low_lap_evidence or patch_blur_evidence or frequency_evidence)
        )
        possible_defocus_evidence = (
            blur_weight >= possible_to_blurry_weight_threshold
            and (low_lap_evidence or low_freq_evidence or low_power_freq_evidence)
            and (patch_blur_evidence or edge_width_absolute_evidence or edge_width_relative_evidence)
        )

        if strong_motion_blur_evidence or soft_motion_blur_evidence:
            row["label"] = "blurry"
            row["blur_type"] = "motion_blur"
        elif defocus_blur_evidence:
            row["label"] = "blurry"
            row["blur_type"] = "defocus_blur"
        elif strong_blur_evidence:
            row["label"] = "blurry"
            if possible_motion_evidence:
                row["blur_type"] = "motion_blur"
            elif possible_defocus_evidence:
                row["blur_type"] = "defocus_blur"
            else:
                row["blur_type"] = "blur_unknown"
        elif possible_motion_evidence:
            row["label"] = "blurry"
            row["blur_type"] = "motion_blur"
        elif possible_defocus_evidence:
            row["label"] = "blurry"
            row["blur_type"] = "defocus_blur"
        else:
            row["label"] = "sharp"
            row["blur_type"] = "none"

    rows = apply_blur_effect_motion_policy_rows(rows)

    for row in rows:
        row["score_min"] = score_min
        row["score_max"] = score_max
        row["patch_blur_threshold"] = patch_blur_th
        row["median_edge_width"] = median_edge_width
    return rows


def enforce_motion_blur_by_weight_rows(rows):
    blurry_rows = [
        row
        for row in rows
        if row.get("label") == "blurry" and not is_missing(row.get("blur_weight"))
    ]
    if len(blurry_rows) == 0:
        return rows

    motion_weight_threshold = float(min(row["blur_weight"] for row in blurry_rows))
    for row in rows:
        blur_weight = row.get("blur_weight")
        if not is_missing(blur_weight) and blur_weight >= motion_weight_threshold:
            row["label"] = "blurry"
            row["blur_type"] = "motion_blur"
            row["motion_weight_threshold"] = motion_weight_threshold
    return rows


def analyze_image_path(path):
    p = Path(path)
    try:
        return analyze_image(p)
    except Exception as e:
        if isinstance(e, ModuleNotFoundError) and e.name == "skimage":
            raise
        return {
            "path": str(p),
            "filename": p.name,
            "valid_patch_count": 0,
            "lap_scores_list": [],
            "lap_p10": np.nan,
            "lap_p20": np.nan,
            "lap_median": np.nan,
            "lap_p80": np.nan,
            "tenengrad_p20": np.nan,
            "tenengrad_median": np.nan,
            "directionality": np.nan,
            "fft_high_ratio": np.nan,
            "fft_power_high_ratio": np.nan,
            "edge_width_median": np.nan,
            "edge_width_p75": np.nan,
            "edge_count": 0,
            "noise": np.nan,
            "raw_score": np.nan,
            "blur_effect": np.nan,
            "blur_effect_label": "uncertain",
            "error": str(e),
        }


def analyze_image_paths(
    image_paths,
    motion_directionality_threshold=1.35,
    motion_soft_directionality_threshold=1.08,
    blurry_patch_percentile=35,
    blurry_patch_ratio_threshold=0.34,
    edge_width_absolute_threshold=10.0,
    motion_blur_weight_threshold=0.30,
    workers=None,
    progress=False,
):
    image_paths = [Path(path) for path in image_paths]
    worker_count = resolve_worker_count(workers, len(image_paths))

    if worker_count <= 1:
        iterator = image_paths
        if progress:
            iterator = tqdm(iterator, desc="Analyzing images")
        rows = [analyze_image_path(path) for path in iterator]
    else:
        rows = [None] * len(image_paths)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(analyze_image_path, path): index
                for index, path in enumerate(image_paths)
            }
            completed = as_completed(futures)
            if progress:
                completed = tqdm(completed, total=len(futures), desc=f"Analyzing images ({worker_count} workers)")
            for future in completed:
                rows[futures[future]] = future.result()

    rows = classify_result_rows(
        rows,
        motion_directionality_threshold=motion_directionality_threshold,
        motion_soft_directionality_threshold=motion_soft_directionality_threshold,
        blurry_patch_percentile=blurry_patch_percentile,
        blurry_patch_ratio_threshold=blurry_patch_ratio_threshold,
        edge_width_absolute_threshold=edge_width_absolute_threshold,
        motion_blur_weight_threshold=motion_blur_weight_threshold,
    )
    for row in rows:
        row["normalized_blur_type"] = normalize_detected_blur_type(row)
    return rows


def detect_image_blur_types(image_dir, recursive=True, **kwargs):
    return analyze_image_paths(collect_images(image_dir, recursive=recursive), **kwargs)


def format_cli_value(value):
    if is_missing(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def classify_results(
    df,
    motion_directionality_threshold=1.35,
    motion_soft_directionality_threshold=1.08,
    blurry_patch_percentile=35,
    blurry_patch_ratio_threshold=0.34,
    edge_width_absolute_threshold=10.0,
    motion_blur_weight_threshold=0.30,

    # 新增：保护清晰图
    clear_force_weight_threshold=0.22,
    clear_force_lap_p20_threshold=120.0,
    clear_force_patch_ratio_threshold=0.12,

    # 新增：possible 转 blurry 必须达到这个严重度
    possible_to_blurry_weight_threshold=0.28,
):
    rows = classify_result_rows(
        df.to_dict("records"),
        motion_directionality_threshold=motion_directionality_threshold,
        motion_soft_directionality_threshold=motion_soft_directionality_threshold,
        blurry_patch_percentile=blurry_patch_percentile,
        blurry_patch_ratio_threshold=blurry_patch_ratio_threshold,
        edge_width_absolute_threshold=edge_width_absolute_threshold,
        motion_blur_weight_threshold=motion_blur_weight_threshold,
        clear_force_weight_threshold=clear_force_weight_threshold,
        clear_force_lap_p20_threshold=clear_force_lap_p20_threshold,
        clear_force_patch_ratio_threshold=clear_force_patch_ratio_threshold,
        possible_to_blurry_weight_threshold=possible_to_blurry_weight_threshold,
    )
    for key in (
        "label",
        "blur_type",
        "blur_weight",
        "blurry_patch_ratio",
        "score_min",
        "score_max",
        "patch_blur_threshold",
        "median_edge_width",
        "blur_effect",
        "blur_effect_label",
        "original_label",
        "original_blur_type",
        "blur_policy",
    ):
        df[key] = [row.get(key, np.nan) for row in rows]
    return df

def enforce_motion_blur_by_weight(df):
    """
    后处理规则：
    blur_weight 越大越模糊。

    如果当前已经有图片被判为 blurry，
    则取 blurry 图片里最小的 blur_weight 作为阈值。
    所有 blur_weight >= 这个阈值的图片，都强制判为：
        label = blurry
        blur_type = motion_blur

    用途：
    避免 blur_weight 更高的图片反而被判成 sharp。
    """

    if "label" not in df.columns or "blur_weight" not in df.columns:
        return df

    blurry_df = df[
        (df["label"] == "blurry")
        & df["blur_weight"].notna()
    ]

    if len(blurry_df) == 0:
        return df

    motion_weight_threshold = float(blurry_df["blur_weight"].min())

    mask = (
        df["blur_weight"].notna()
        & (df["blur_weight"] >= motion_weight_threshold)
    )

    df.loc[mask, "label"] = "blurry"
    df.loc[mask, "blur_type"] = "motion_blur"
    df.loc[mask, "motion_weight_threshold"] = motion_weight_threshold

    return df

def collect_images(image_dir, recursive=True):
    image_dir = Path(image_dir)

    if recursive:
        paths = [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    else:
        paths = [
            p for p in image_dir.glob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Detect blurry images and estimate blur type."
    )

    parser.add_argument(
        "image_dir",
        type=str,
        help="Folder containing images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="blur_results.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only read images directly under image_dir, not subfolders.",
    )

    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=1.35,
        help="Strong directionality threshold for motion blur. Default: 1.35",
    )

    parser.add_argument(
        "--motion-soft-threshold",
        type=float,
        default=1.08,
        help="Soft directionality threshold for weak motion blur. Default: 1.08",
    )

    parser.add_argument(
        "--motion-blur-weight-threshold",
        type=float,
        default=0.30,
        help="Blur weight threshold for soft motion blur. Default: 0.30",
    )

    parser.add_argument(
        "--patch-blur-percentile",
        type=float,
        default=35,
        help="Patch Laplacian percentile used as blur threshold. Default: 35",
    )

    parser.add_argument(
        "--patch-ratio-threshold",
        type=float,
        default=0.34,
        help="If this ratio of patches are blurry, image is blurry. Default: 0.34",
    )

    parser.add_argument(
        "--edge-width-threshold",
        type=float,
        default=10.0,
        help="Absolute edge spread width threshold. Default: 10.0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel image analysis workers. Use 0 for auto, 1 for serial. Default: 0",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")

    image_paths = collect_images(
        image_dir,
        recursive=not args.no_recursive,
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    rows = analyze_image_paths(
        image_paths,
        motion_directionality_threshold=args.motion_threshold,
        motion_soft_directionality_threshold=args.motion_soft_threshold,
        blurry_patch_percentile=args.patch_blur_percentile,
        blurry_patch_ratio_threshold=args.patch_ratio_threshold,
        edge_width_absolute_threshold=args.edge_width_threshold,
        motion_blur_weight_threshold=args.motion_blur_weight_threshold,
        workers=args.workers,
        progress=True,
    )

    rows = sorted(
        rows,
        key=lambda row: (
            -(float(row["blur_weight"]) if not is_missing(row.get("blur_weight")) else -1.0),
            float(row["raw_score"]) if not is_missing(row.get("raw_score")) else float("inf"),
        ),
    )

    output_path = Path(args.output)
    show_cols = [
        "filename",
        "label",
        "blur_type",
        "blur_weight",
        "blur_effect",
        "blur_effect_label",
        "original_blur_type",
        "blurry_patch_ratio",
        "raw_score",
        "lap_p20",
        "edge_width_median",
        "directionality",
        "fft_power_high_ratio",
        "fft_high_ratio",
        "valid_patch_count",
    ]
    fieldnames = list(dict.fromkeys(["path", *show_cols, "normalized_blur_type", "error"]))
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Blur Detection Results ===")
    print(" | ".join(show_cols))
    for row in rows:
        print(" | ".join(format_cli_value(row.get(col)) for col in show_cols))

    print(f"\nSaved results to: {output_path.resolve()}")

    print("\n=== Blurry Images ===")
    blurry_rows = [row for row in rows if row.get("label") == "blurry"]

    if len(blurry_rows) == 0:
        print("No blurry images detected.")
    else:
        for row in blurry_rows:
            print(
                f"{row['filename']} | "
                f"{row['blur_type']} | "
                f"weight={format_cli_value(row.get('blur_weight'))} | "
                f"patch_ratio={format_cli_value(row.get('blurry_patch_ratio'))} | "
                f"edge_width={format_cli_value(row.get('edge_width_median'))} | "
                f"directionality={format_cli_value(row.get('directionality'))} | "
                f"raw_score={format_cli_value(row.get('raw_score'))}"
            )
    print("\n=== Sharp Images ===")
    sharp_rows = [row for row in rows if row.get("label") == "sharp"]

    if len(sharp_rows) == 0:
        print("No sharp images detected.")
    else:
        for row in sharp_rows:
            print(row["filename"])
    print("\n=== Suggested Deblur Usage ===")
    print("Deblur:")
    print("  deblur = label == 'blurry'")
    print("")
    print("Motion deblur:")
    print("  deblur = label == 'blurry' and blur_type == 'motion_blur'")
    print("")
    print("Defocus deblur:")
    print("  deblur = label == 'blurry' and blur_type == 'defocus_blur'")


if __name__ == "__main__":
    main()
