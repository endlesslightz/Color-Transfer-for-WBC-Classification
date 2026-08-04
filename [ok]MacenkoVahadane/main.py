"""Run Macenko and Vahadane color transfer against a reference image.

Usage:
    python main.py
    python main.py --method both --device cuda
    python main.py --method macenko --device cpu
    python main.py --method vahadane --device auto
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch

from stain_normalization import MacenkoNormalizer, VahadaneNormalizer
from stain_normalization.torch_backend import (
    dictionary_backend,
    has_spams,
    resolve_device,
)
from stain_normalization.utils import list_images, load_rgb, save_rgb

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "images"
DEFAULT_REFERENCE = DEFAULT_IMAGES / "may-grunwald.jpg"
DEFAULT_OUTPUT = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stain color transfer with Macenko and/or Vahadane."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help=(
            "Root folder to scan for source images in subfolders "
            "(default: ./images; reference stays at images/may-grunwald.jpg)"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference image whose stain nuance is transferred",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root output folder (default: ./output)",
    )
    parser.add_argument(
        "--method",
        choices=("macenko", "vahadane", "both"),
        default="both",
        help="Which normalizer(s) to run",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device (default: auto -> cuda if available else cpu)",
    )
    return parser.parse_args()


def _gpu_memory_mb(*, use_torch_peak: bool) -> float | None:
    """Return GPU memory usage in MiB, or None if unavailable."""
    if use_torch_peak:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


def _reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def process_batch(
    method_name: str,
    normalizer,
    source_paths: list[Path],
    source_root: Path,
    reference_rgb,
    output_dir: Path,
) -> None:
    device = normalizer.device
    print(f"[{method_name}] Device: {device}")
    print(f"[{method_name}] Fitting reference stain appearance...")
    normalizer.fit(reference_rgb)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = device.type == "cuda"
    if use_cuda:
        _reset_gpu_peak()
    mem_before = _gpu_memory_mb(use_torch_peak=use_cuda) if use_cuda else None
    peak_gpu_mb = mem_before
    times_s: list[float] = []

    for source_path in source_paths:
        rel = source_path.relative_to(source_root)
        print(f"[{method_name}] Processing {rel}...")

        t0 = time.perf_counter()
        source_rgb = load_rgb(source_path)
        result = normalizer.transform(source_rgb)
        if use_cuda:
            torch.cuda.synchronize()
        out_path = output_dir / rel
        save_rgb(out_path, result)
        elapsed = time.perf_counter() - t0
        times_s.append(elapsed)

        if use_cuda:
            mem_now = _gpu_memory_mb(use_torch_peak=True)
            if mem_now is not None:
                peak_gpu_mb = mem_now if peak_gpu_mb is None else max(peak_gpu_mb, mem_now)

        print(f"[{method_name}] Saved -> {out_path} ({elapsed:.3f}s)")

    avg_s = sum(times_s) / len(times_s) if times_s else 0.0
    print(f"[{method_name}] --- Performance ---")
    print(f"[{method_name}] Images processed          : {len(times_s)}")
    print(f"[{method_name}] Average Processing Time   : {avg_s:.3f} s/image")
    if not use_cuda:
        print(f"[{method_name}] GPU Memory Consumption    : N/A (device={device})")
    elif peak_gpu_mb is None:
        print(f"[{method_name}] GPU Memory Consumption    : N/A")
    else:
        delta = peak_gpu_mb - mem_before if mem_before is not None else peak_gpu_mb
        print(
            f"[{method_name}] GPU Memory Consumption    : "
            f"peak {peak_gpu_mb:.1f} MiB (delta {delta:+.1f} MiB)"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(
        "Method: reference-aligned Macenko/Vahadane "
        "(brightness standardize, OD=-log(I/255), Vahadane without maxC scaling)"
    )
    print(
        f"Vahadane dictionary: {dictionary_backend()}"
        + (" (SPAMS available)" if has_spams() else " (SPAMS not available on this platform)")
    )

    if not args.reference.exists():
        raise FileNotFoundError(
            f"Reference image not found: {args.reference}\n"
            "Place may-grunwald.jpg under images/ or pass --reference."
        )

    reference_rgb = load_rgb(args.reference)
    source_root = args.images.resolve()
    source_paths = list_images(
        source_root,
        recursive=True,
        exclude_paths={args.reference.resolve()},
    )

    if not source_paths:
        print(
            f"No source images found under {args.images} "
            f"(excluding reference '{args.reference.name}').\n"
            "Add histology images to subfolders under images/ (e.g. Train/, Val/), then re-run."
        )
        # Still demonstrate the pipeline by normalizing the reference onto itself.
        source_paths = [args.reference]
        print(f"Demo fallback: normalizing reference onto itself -> {args.reference.name}")

    if args.method in ("macenko", "both"):
        process_batch(
            "Macenko",
            MacenkoNormalizer(device=device),
            source_paths,
            source_root,
            reference_rgb,
            args.output / "macenko",
        )

    if args.method in ("vahadane", "both"):
        process_batch(
            "Vahadane",
            VahadaneNormalizer(device=device),
            source_paths,
            source_root,
            reference_rgb,
            args.output / "vahadane",
        )

    print("Done.")


if __name__ == "__main__":
    main()
