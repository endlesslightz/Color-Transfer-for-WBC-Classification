"""Macenko stain normalization (reference math, PyTorch GPU/CPU runtime).

Aligned with stainNorm_Macenko.py / Macenko et al., ISBI 2009.
"""

from __future__ import annotations

import numpy as np
import torch

from .torch_backend import (
    as_image_tensor,
    default_dtype,
    get_concentrations,
    normalize_rows,
    od_to_rgb,
    resolve_device,
    rgb_to_od,
    standardize_brightness,
)


def get_stain_matrix(
    image: torch.Tensor,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Estimate 2x3 stain matrix (rows = stains) from an RGB tensor."""
    od = rgb_to_od(image).reshape((-1, 3))
    tissue = torch.any(od > beta, dim=1)
    od_tissue = od[tissue]
    if od_tissue.shape[0] < 10:
        raise ValueError("Too few tissue pixels to estimate Macenko stains.")

    # Covariance + eigh (same as stainNorm_Macenko), then take 2 largest eigenvectors.
    # cov: centered Gram / (n-1)
    n = od_tissue.shape[0]
    mean = od_tissue.mean(dim=0, keepdim=True)
    centered = od_tissue - mean
    cov = (centered.T @ centered) / max(n - 1, 1)
    _, eigenvectors = torch.linalg.eigh(cov)
    V = eigenvectors[:, [2, 1]].contiguous()

    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1

    projection = od_tissue @ V
    phi = torch.atan2(projection[:, 1], projection[:, 0])
    min_phi = torch.quantile(phi.float(), alpha / 100.0).to(dtype=image.dtype)
    max_phi = torch.quantile(phi.float(), (100.0 - alpha) / 100.0).to(dtype=image.dtype)

    v1 = V @ torch.stack([torch.cos(min_phi), torch.sin(min_phi)])
    v2 = V @ torch.stack([torch.cos(max_phi), torch.sin(max_phi)])

    if v1[0] > v2[0]:
        he = torch.stack([v1, v2], dim=0)
    else:
        he = torch.stack([v2, v1], dim=0)
    return normalize_rows(he)


class MacenkoNormalizer:
    """Color-transfer normalizer based on the Macenko algorithm (GPU-capable)."""

    def __init__(
        self,
        beta: float = 0.15,
        alpha: float = 1.0,
        max_conc_percentile: float = 99.0,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.beta = beta
        self.alpha = alpha
        self.max_conc_percentile = max_conc_percentile
        self.device = resolve_device(device)
        self.dtype = default_dtype(self.device)

        self.stain_matrix_target: torch.Tensor | None = None
        self.target_concentrations: torch.Tensor | None = None

    def fit(self, target_rgb: np.ndarray | torch.Tensor) -> "MacenkoNormalizer":
        target = as_image_tensor(target_rgb, self.device, self.dtype)
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(
            target, beta=self.beta, alpha=self.alpha
        )
        od = rgb_to_od(target).reshape((-1, 3))
        self.target_concentrations = get_concentrations(od, self.stain_matrix_target)
        return self

    def transform(self, source_rgb: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.stain_matrix_target is None or self.target_concentrations is None:
            raise RuntimeError("Call fit(reference_image) before transform().")

        image = as_image_tensor(source_rgb, self.device, self.dtype)
        image = standardize_brightness(image)
        h, w = image.shape[:2]

        stain_matrix_source = get_stain_matrix(
            image, beta=self.beta, alpha=self.alpha
        )
        od = rgb_to_od(image).reshape((-1, 3))
        source_concentrations = get_concentrations(od, stain_matrix_source)

        max_c_source = torch.quantile(
            source_concentrations.float(),
            self.max_conc_percentile / 100.0,
            dim=0,
        ).to(dtype=self.dtype).clamp(min=1e-6)
        max_c_target = torch.quantile(
            self.target_concentrations.float(),
            self.max_conc_percentile / 100.0,
            dim=0,
        ).to(dtype=self.dtype)

        source_concentrations = source_concentrations * (max_c_target / max_c_source)
        od_normalized = source_concentrations @ self.stain_matrix_target
        rgb = od_to_rgb(od_normalized.reshape(h, w, 3))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return rgb.detach().cpu().numpy()

    def fit_transform(
        self, source_rgb: np.ndarray | torch.Tensor, target_rgb: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        return self.fit(target_rgb).transform(source_rgb)
