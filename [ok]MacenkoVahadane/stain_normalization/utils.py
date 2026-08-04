"""Shared helpers for optical-density stain normalization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 array (H, W, 3)."""
    path = Path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Save an RGB uint8 array to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"Unable to write image: {path}")


def list_images(
    folder: str | Path,
    *,
    recursive: bool = False,
    exclude_names: set[str] | None = None,
    exclude_paths: set[str | Path] | None = None,
) -> list[Path]:
    """List image files under a folder, optionally recursively."""
    folder = Path(folder)
    exclude_names = exclude_names or set()
    exclude_resolved = {Path(p).resolve() for p in (exclude_paths or set())}

    if recursive:
        candidates = (
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        candidates = (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    files = [
        p
        for p in sorted(candidates)
        if p.name not in exclude_names and p.resolve() not in exclude_resolved
    ]
    return files


def rgb_to_od(image_rgb: np.ndarray, iod: float = 240.0) -> np.ndarray:
    """Convert RGB intensities to optical density via Beer-Lambert law."""
    image = image_rgb.astype(np.float64)
    image = np.maximum(image, 1.0)
    return -np.log((image + 1.0) / (iod + 1.0))


def od_to_rgb(od: np.ndarray, iod: float = 240.0) -> np.ndarray:
    """Convert optical density back to RGB uint8."""
    rgb = (iod + 1.0) * np.exp(-od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row of a 2D matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def normalize_columns(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each column of a 2D matrix."""
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, eps)


def get_tissue_mask(od: np.ndarray, beta: float = 0.15) -> np.ndarray:
    """Boolean mask of tissue pixels (OD above beta in at least one channel)."""
    return np.any(od > beta, axis=1)
