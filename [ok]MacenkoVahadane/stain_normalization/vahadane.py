"""Vahadane stain normalization (reference math, PyTorch GPU/CPU runtime).

Aligned with stainNorm_Vahadane.py / Vahadane et al., IEEE TMI 2016.
Dictionary learning uses GPU SNMF (SPAMS trainDL equivalent path).
"""

from __future__ import annotations

import numpy as np
import torch

from .torch_backend import (
    as_image_tensor,
    default_dtype,
    get_concentrations,
    normalize_rows,
    notwhite_mask_numpy,
    od_to_rgb,
    order_stains_purple_first,
    resolve_device,
    rgb_to_od,
    standardize_brightness,
    train_stain_dictionary,
)


def get_stain_matrix(
    image: torch.Tensor,
    image_u8_cpu: np.ndarray,
    *,
    threshold: float = 0.8,
    lamda: float = 0.1,
    max_iter: int = 100,
    random_state: int = 0,
) -> torch.Tensor:
    """Get 2x3 stain matrix. First row nuclear/purple-like, second residual stain."""
    mask_np = notwhite_mask_numpy(image_u8_cpu, thresh=threshold).reshape((-1,))
    mask = torch.as_tensor(mask_np, device=image.device)
    od = rgb_to_od(image).reshape((-1, 3))
    od_tissue = od[mask]
    if od_tissue.shape[0] < 10:
        raise ValueError("Too few tissue pixels to estimate Vahadane stains.")

    dictionary = train_stain_dictionary(
        od_tissue,
        n_stains=2,
        lambda1=lamda,
        max_iter=max_iter,
        random_state=random_state,
    )
    # MGG-aware ordering (avoids mapping nuclei onto a green stain vector).
    dictionary = order_stains_purple_first(dictionary)
    return normalize_rows(dictionary)


class VahadaneNormalizer:
    """Color-transfer normalizer based on Vahadane (GPU-capable SNMF)."""

    def __init__(
        self,
        threshold: float = 0.8,
        lamda: float = 0.1,
        max_iter: int = 100,
        random_state: int = 0,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.threshold = threshold
        self.lamda = lamda
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = resolve_device(device)
        self.dtype = default_dtype(self.device)

        self.stain_matrix_target: torch.Tensor | None = None

    def fit(self, target_rgb: np.ndarray | torch.Tensor) -> "VahadaneNormalizer":
        target = as_image_tensor(target_rgb, self.device, self.dtype)
        target = standardize_brightness(target)
        target_u8 = target.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()

        self.stain_matrix_target = get_stain_matrix(
            target,
            target_u8,
            threshold=self.threshold,
            lamda=self.lamda,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        return self

    def transform(self, source_rgb: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.stain_matrix_target is None:
            raise RuntimeError("Call fit(reference_image) before transform().")

        image = as_image_tensor(source_rgb, self.device, self.dtype)
        image = standardize_brightness(image)
        h, w = image.shape[:2]
        image_u8 = image.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()

        stain_matrix_source = get_stain_matrix(
            image,
            image_u8,
            threshold=self.threshold,
            lamda=self.lamda,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        od = rgb_to_od(image).reshape((-1, 3))
        source_concentrations = get_concentrations(od, stain_matrix_source)
        # Reference Vahadane: no max-concentration scaling.
        od_normalized = source_concentrations @ self.stain_matrix_target
        rgb = od_to_rgb(od_normalized.reshape(h, w, 3))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return rgb.detach().cpu().numpy()

    def fit_transform(
        self, source_rgb: np.ndarray | torch.Tensor, target_rgb: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        return self.fit(target_rgb).transform(source_rgb)
