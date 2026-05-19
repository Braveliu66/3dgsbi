import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


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


def analyze_image(path):
    gray = read_gray_image(path)

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
    }


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
    valid = df["raw_score"].notna()
    scores = df.loc[valid, "raw_score"].values

    if len(scores) == 0:
        df["label"] = "uncertain"
        df["blur_type"] = "uncertain"
        df["blur_weight"] = 0.5
        df["blurry_patch_ratio"] = np.nan
        return df

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))

    median_fft = df["fft_high_ratio"].median(skipna=True)
    median_power_fft = df["fft_power_high_ratio"].median(skipna=True)
    median_lap = df["lap_p20"].median(skipna=True)
    median_edge_width = df["edge_width_median"].median(skipna=True)

    all_patch_laps = []
    for v in df["lap_scores_list"]:
        if isinstance(v, list):
            all_patch_laps.extend(v)

    if len(all_patch_laps) > 0:
        patch_blur_th = float(np.percentile(all_patch_laps, blurry_patch_percentile))
    else:
        patch_blur_th = median_lap

    labels = []
    blur_types = []
    weights = []
    blurry_patch_ratios = []

    for _, row in df.iterrows():
        s = row.get("raw_score", np.nan)

        if pd.isna(s):
            labels.append("uncertain")
            blur_types.append("uncertain")
            weights.append(0.5)
            blurry_patch_ratios.append(np.nan)
            continue

        blur_weight = (score_max - s) / max(score_max - score_min, 1e-6)
        blur_weight = float(np.clip(blur_weight, 0.0, 1.0))
        weights.append(blur_weight)

        directionality = row.get("directionality", np.nan)
        fft_high = row.get("fft_high_ratio", np.nan)
        power_fft = row.get("fft_power_high_ratio", np.nan)
        lap_p20 = row.get("lap_p20", np.nan)
        edge_width = row.get("edge_width_median", np.nan)

        patch_laps = row.get("lap_scores_list", [])

        if isinstance(patch_laps, list) and len(patch_laps) > 0 and not pd.isna(patch_blur_th):
            blurry_patch_ratio = float(np.mean(np.array(patch_laps) < patch_blur_th))
        else:
            blurry_patch_ratio = np.nan

        blurry_patch_ratios.append(blurry_patch_ratio)

        is_strong_directional = (
            not pd.isna(directionality)
            and directionality >= motion_directionality_threshold
        )

        is_soft_directional = (
            not pd.isna(directionality)
            and directionality >= motion_soft_directionality_threshold
        )

        low_freq_evidence = (
            not pd.isna(fft_high)
            and not pd.isna(median_fft)
            and fft_high < median_fft
        )

        low_power_freq_evidence = (
            not pd.isna(power_fft)
            and not pd.isna(median_power_fft)
            and power_fft < median_power_fft
        )

        low_lap_evidence = (
            not pd.isna(lap_p20)
            and not pd.isna(median_lap)
            and lap_p20 < median_lap
        )

        patch_blur_evidence = (
            not pd.isna(blurry_patch_ratio)
            and blurry_patch_ratio >= blurry_patch_ratio_threshold
        )

        edge_width_absolute_evidence = (
            not pd.isna(edge_width)
            and edge_width >= edge_width_absolute_threshold
        )

        edge_width_relative_evidence = (
            not pd.isna(edge_width)
            and not pd.isna(median_edge_width)
            and edge_width >= median_edge_width * 1.20
        )

        frequency_evidence = low_freq_evidence or low_power_freq_evidence

        # 新增：强制保护清晰图
        # 对应你这批 007/035/000/028/042/014/021 这种情况
        clear_force_evidence = (
            blur_weight <= clear_force_weight_threshold
            and not pd.isna(lap_p20)
            and lap_p20 >= clear_force_lap_p20_threshold
            and not pd.isna(blurry_patch_ratio)
            and blurry_patch_ratio <= clear_force_patch_ratio_threshold
        )

        if clear_force_evidence:
            labels.append("sharp")
            blur_types.append("none")
            continue

        strong_motion_blur_evidence = (
            is_strong_directional
            and (
                edge_width_absolute_evidence
                or (patch_blur_evidence and frequency_evidence)
            )
        )

        # 004/005 主要靠这个判据识别
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
            and (
                low_power_freq_evidence
                or patch_blur_evidence
                or low_lap_evidence
            )
        )

        # possible 不再单独把清晰图打成 blurry
        # 必须 blur_weight 足够高，并且至少有低 Laplacian / 低频 / patch 证据
        possible_motion_evidence = (
            blur_weight >= possible_to_blurry_weight_threshold
            and is_soft_directional
            and edge_width_absolute_evidence
            and (low_lap_evidence or patch_blur_evidence or frequency_evidence)
        )

        possible_defocus_evidence = (
            blur_weight >= possible_to_blurry_weight_threshold
            and (
                low_lap_evidence
                or low_freq_evidence
                or low_power_freq_evidence
            )
            and (
                patch_blur_evidence
                or edge_width_absolute_evidence
                or edge_width_relative_evidence
            )
        )

        if strong_motion_blur_evidence or soft_motion_blur_evidence:
            labels.append("blurry")
            blur_types.append("motion_blur")

        elif defocus_blur_evidence:
            labels.append("blurry")
            blur_types.append("defocus_blur")

        elif strong_blur_evidence:
            labels.append("blurry")

            if possible_motion_evidence:
                blur_types.append("motion_blur")
            elif possible_defocus_evidence:
                blur_types.append("defocus_blur")
            else:
                blur_types.append("blur_unknown")

        elif possible_motion_evidence:
            labels.append("blurry")
            blur_types.append("motion_blur")

        elif possible_defocus_evidence:
            labels.append("blurry")
            blur_types.append("defocus_blur")

        else:
            labels.append("sharp")
            blur_types.append("none")

    df["label"] = labels
    df["blur_type"] = blur_types
    df["blur_weight"] = weights
    df["blurry_patch_ratio"] = blurry_patch_ratios
    df["score_min"] = score_min
    df["score_max"] = score_max
    df["patch_blur_threshold"] = patch_blur_th
    df["median_edge_width"] = median_edge_width

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

    rows = []

    for p in tqdm(image_paths, desc="Analyzing images"):
        try:
            rows.append(analyze_image(p))
        except Exception as e:
            rows.append({
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
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    df = classify_results(
        df,
        motion_directionality_threshold=args.motion_threshold,
        motion_soft_directionality_threshold=args.motion_soft_threshold,
        blurry_patch_percentile=args.patch_blur_percentile,
        blurry_patch_ratio_threshold=args.patch_ratio_threshold,
        edge_width_absolute_threshold=args.edge_width_threshold,
        motion_blur_weight_threshold=args.motion_blur_weight_threshold,
    )

    df = enforce_motion_blur_by_weight(df)

    df = df.sort_values(
        ["blur_weight", "raw_score"],
        ascending=[False, True],
    )

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    show_cols = [
        "filename",
        "label",
        "blur_type",
        "blur_weight",
        "blurry_patch_ratio",
        "raw_score",
        "lap_p20",
        "edge_width_median",
        "directionality",
        "fft_power_high_ratio",
        "fft_high_ratio",
        "valid_patch_count",
    ]

    print("\n=== Blur Detection Results ===")
    print(df[show_cols].to_string(index=False))

    print(f"\nSaved results to: {output_path.resolve()}")

    print("\n=== Blurry Images ===")
    blurry_df = df[df["label"] == "blurry"]

    if len(blurry_df) == 0:
        print("No blurry images detected.")
    else:
        for _, row in blurry_df.iterrows():
            print(
                f"{row['filename']} | "
                f"{row['blur_type']} | "
                f"weight={row['blur_weight']:.3f} | "
                f"patch_ratio={row['blurry_patch_ratio']:.3f} | "
                f"edge_width={row['edge_width_median']:.2f} | "
                f"directionality={row['directionality']:.3f} | "
                f"raw_score={row['raw_score']:.3f}"
            )
    print("\n=== Sharp Images ===")
    sharp_df = df[df["label"] == "sharp"]

    if len(sharp_df) == 0:
        print("No sharp images detected.")
    else:
        for filename in sharp_df["filename"].tolist():
            print(filename)
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