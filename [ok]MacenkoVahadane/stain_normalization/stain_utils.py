"""Helpers aligned with classic stainNorm / staintools stain_utils."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import spams

    _HAS_SPAMS = True
except ImportError:
    spams = None
    _HAS_SPAMS = False


def has_spams() -> bool:
    return _HAS_SPAMS


def remove_zeros(I: np.ndarray) -> np.ndarray:
    """Replace zeros with ones to avoid log(0)."""
    out = I.astype(np.float64, copy=True)
    out[out == 0] = 1
    return out


def standardize_brightness(I: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Scale image so the given intensity percentile maps to 255.

    Matches classic mitkovetta/stainNorm ``stain_utils.standardize_brightness``.
    """
    I = np.asarray(I)
    p = np.percentile(I, percentile)
    if p <= 0:
        return I.astype(np.uint8, copy=False)
    return np.clip(I.astype(np.float64) * 255.0 / p, 0, 255).astype(np.uint8)


def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to optical density: OD = -log(I / 255)."""
    I = remove_zeros(np.asarray(I, dtype=np.float64))
    return -np.log(I / 255.0)


def OD_to_RGB(OD: np.ndarray) -> np.ndarray:
    """Convert optical density back to RGB uint8: RGB = 255 * exp(-OD)."""
    return (255.0 * np.exp(-np.asarray(OD, dtype=np.float64))).clip(0, 255).astype(np.uint8)


def normalize_rows(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(norms, eps)


def notwhite_mask(I: np.ndarray, thresh: float = 0.8) -> np.ndarray:
    """Tissue mask: LAB luminance below ``thresh`` (reference Vahadane default 0.8)."""
    if cv2 is None:
        raise ImportError("opencv-python is required for notwhite_mask")
    I_u8 = np.asarray(I, dtype=np.uint8)
    lab = cv2.cvtColor(I_u8, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float64) / 255.0
    return L < thresh


def get_concentrations(
    I: np.ndarray,
    stain_matrix: np.ndarray,
    lamda: float = 0.01,
) -> np.ndarray:
    """Estimate stain concentrations (N x 2).

    Prefer SPAMS lasso (reference). Fallback: non-negative least squares per pixel
    batch-solved via clipped least squares (fast, approximate).
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    stain_matrix = np.asarray(stain_matrix, dtype=np.float64)

    if _HAS_SPAMS:
        # spams.lasso expects D shape (n_features, n_atoms) = (3, 2)
        coef = spams.lasso(
            np.asfortranarray(OD.T),
            D=np.asfortranarray(stain_matrix.T),
            mode=2,
            lambda1=lamda,
            pos=True,
        )
        return np.asarray(coef.toarray().T, dtype=np.float64)

    # Fallback: solve stain_matrix.T @ c ~= od  i.e. (3x2) @ (2,) ~= (3,)
    # Batch lstsq: stain_matrix.T (3x2), OD.T (3xN) -> (2xN), then clip.
    concentrations = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)[0].T
    return np.maximum(concentrations, 0.0)


def train_stain_dictionary(
    OD_tissue: np.ndarray,
    *,
    n_stains: int = 2,
    lambda1: float = 0.1,
    max_iter: int = 100,
    random_state: int = 0,
) -> np.ndarray:
    """Learn stain dictionary; returns matrix shaped (n_stains, 3).

    Prefer ``spams.trainDL`` (reference Vahadane). Fallback: sparse NMF.
    ``OD_tissue`` is (N_tissue, 3).
    """
    OD_tissue = np.asarray(OD_tissue, dtype=np.float64)
    if OD_tissue.ndim != 2 or OD_tissue.shape[1] != 3:
        raise ValueError("OD_tissue must have shape (N, 3)")

    if _HAS_SPAMS:
        dictionary = spams.trainDL(
            np.asfortranarray(OD_tissue.T),
            K=n_stains,
            lambda1=lambda1,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False,
        ).T
        return np.asarray(dictionary, dtype=np.float64)

    return _snmf_dictionary(
        OD_tissue.T,
        n_stains=n_stains,
        lambda1=lambda1,
        max_iter=max_iter,
        random_state=random_state,
    )


def _snmf_dictionary(
    od_matrix: np.ndarray,
    *,
    n_stains: int,
    lambda1: float,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    """Multiplicative-update SNMF fallback. ``od_matrix`` is (3, N). Returns (K, 3)."""
    rng = np.random.default_rng(random_state)
    n_channels, n_pixels = od_matrix.shape
    k = n_stains

    w = rng.random((n_channels, k)) + 1e-3
    h = rng.random((k, n_pixels)) + 1e-3
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-8)

    eps = 1e-8
    for _ in range(max_iter):
        numerator_h = w.T @ od_matrix
        denominator_h = (w.T @ w) @ h + lambda1 + eps
        h *= numerator_h / denominator_h
        h = np.maximum(h, 0.0)

        numerator_w = od_matrix @ h.T
        denominator_w = w @ (h @ h.T) + eps
        w *= numerator_w / denominator_w
        w = np.maximum(w, 0.0)
        w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-8)

    return w.T  # (K, 3)
