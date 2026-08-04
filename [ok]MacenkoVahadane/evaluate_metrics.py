"""Evaluate stain-normalized outputs.

For each source under ./images (excluding may-grunwald.jpg) and its pair in
./output/<method>/:
  - RMSE, PSNR, SSIM : output vs reference (may-grunwald.jpg, resized to output)
  - CTQM             : source, reference, output

Usage:
    python evaluate_metrics.py
    python evaluate_metrics.py --method macenko
    python evaluate_metrics.py --method vahadane
    python evaluate_metrics.py --method both
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from CTQM import ctqm
from metrics import psnr, rmse, ssim
from stain_normalization.utils import list_images

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "images"
DEFAULT_REFERENCE = DEFAULT_IMAGES / "may-grunwald.jpg"
DEFAULT_OUTPUT = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare source images with normalized outputs "
            "(PSNR, RMSE, SSIM, CTQM)."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Source images root (default: ./images)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference stain image used for CTQM (default: images/may-grunwald.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Normalized output root containing macenko/ and vahadane/ (default: ./output)",
    )
    parser.add_argument(
        "--method",
        choices=("macenko", "vahadane", "both"),
        default="both",
        help="Which output method folder(s) to evaluate",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "CSV path for detailed results. Default: "
            "evaluation_metrics_<method>.csv in the project root"
        ),
    )
    return parser.parse_args()


def _load_bgr(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _resize_to_match(image: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resize HxW to match target (H, W, C) if needed."""
    th, tw = target_shape[:2]
    if image.shape[0] == th and image.shape[1] == tw:
        return image
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)


def evaluate_method(
    method_name: str,
    source_paths: list[Path],
    source_root: Path,
    output_root: Path,
    reference_bgr: np.ndarray,
) -> pd.DataFrame:
    method_dir = output_root / method_name
    if not method_dir.is_dir():
        print(f"[{method_name}] Output folder not found: {method_dir}")
        return pd.DataFrame()

    results: list[dict] = []
    skipped_missing = 0
    skipped_unreadable = 0

    print(f"[{method_name}] Evaluating against {method_dir} ...")
    for src_path in tqdm(source_paths, desc=f"Metrics [{method_name}]"):
        rel = src_path.relative_to(source_root)
        out_path = method_dir / rel
        if not out_path.exists():
            skipped_missing += 1
            continue

        src_bgr = _load_bgr(src_path)
        out_bgr = _load_bgr(out_path)
        if src_bgr is None or out_bgr is None:
            skipped_unreadable += 1
            continue

        src_matched = _resize_to_match(src_bgr, out_bgr.shape)
        ref_matched = _resize_to_match(reference_bgr, out_bgr.shape)

        # RMSE / PSNR / SSIM: output vs reference (same as reference eval script).
        val_rmse = rmse(out_bgr, ref_matched)
        val_psnr = psnr(out_bgr, ref_matched)
        val_ssim = ssim(out_bgr, ref_matched)

        # CTQM: source, reference, output (RGB float [0, 1]).
        src_rgb_01 = cv2.cvtColor(src_matched, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_rgb_01 = cv2.cvtColor(ref_matched, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_rgb_01 = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        try:
            val_ctqm = ctqm(src_rgb_01, ref_rgb_01, out_rgb_01)
        except Exception:
            val_ctqm = np.nan

        results.append(
            {
                "method": method_name,
                "relative_path": str(rel).replace("\\", "/"),
                "RMSE": val_rmse,
                "PSNR": val_psnr,
                "SSIM": val_ssim,
                "CTQM": val_ctqm,
            }
        )

        if len(results) % 50 == 0:
            gc.collect()

    if skipped_missing:
        print(f"[{method_name}] Skipped (missing output): {skipped_missing}")
    if skipped_unreadable:
        print(f"[{method_name}] Skipped (unreadable): {skipped_unreadable}")

    return pd.DataFrame(results)


def print_summary(method_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"[{method_name}] No pairs evaluated.")
        return

    print(f"\n=== [{method_name}] HASIL RATA-RATA METRIK (n={len(df)}) ===")
    print(f"RMSE : {df['RMSE'].mean():.4f}")
    print(f"PSNR : {df['PSNR'].mean():.4f} dB")
    print(f"SSIM : {df['SSIM'].mean():.4f}")
    print(f"CTQM : {df['CTQM'].mean():.4f}")


def main() -> None:
    args = parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")

    reference_bgr = _load_bgr(args.reference)
    if reference_bgr is None:
        raise FileNotFoundError(f"Unable to read reference image: {args.reference}")

    source_root = args.images.resolve()
    source_paths = list_images(
        source_root,
        recursive=True,
        exclude_paths={args.reference.resolve()},
    )

    if not source_paths:
        print(f"No source images found under {args.images} (excluding reference).")
        return

    methods = ("macenko", "vahadane") if args.method == "both" else (args.method,)
    all_frames: list[pd.DataFrame] = []

    print(f"Source images: {len(source_paths)} under {source_root}")
    print(f"Reference    : {args.reference}")

    for method in methods:
        df = evaluate_method(
            method,
            source_paths,
            source_root,
            args.output.resolve(),
            reference_bgr,
        )
        print_summary(method, df)
        if not df.empty:
            all_frames.append(df)
            csv_path = args.csv
            if csv_path is None:
                csv_path = ROOT / f"evaluation_metrics_{method}.csv"
            elif args.method == "both":
                csv_path = csv_path.with_name(
                    f"{csv_path.stem}_{method}{csv_path.suffix}"
                )
            df.to_csv(csv_path, index=False)
            print(f"[{method}] Detail disimpan ke '{csv_path}'")

    if len(all_frames) > 1:
        print("\n=== RINGKASAN AKHIR SEMUA METHOD ===")
        for df in all_frames:
            method = df["method"].iloc[0]
            print(
                f"{method:8s}  n={len(df):4d}  "
                f"RMSE={df['RMSE'].mean():.4f}  "
                f"PSNR={df['PSNR'].mean():.4f}  "
                f"SSIM={df['SSIM'].mean():.4f}  "
                f"CTQM={df['CTQM'].mean():.4f}"
            )


if __name__ == "__main__":
    main()
