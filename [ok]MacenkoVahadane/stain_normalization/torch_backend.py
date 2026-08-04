"""PyTorch helpers for GPU/CPU stain normalization (reference-aligned math).

When SPAMS is installed, Vahadane dictionary learning and sparse concentrations
use SPAMS (CPU) exactly like stainNorm_Vahadane; otherwise SNMF/lstsq on device.
"""

from __future__ import annotations

import numpy as np
import torch

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


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve 'cuda' / 'cpu' / 'auto' / None to a torch.device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        resolved = device
    else:
        resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch or use --device cpu."
        )
    return resolved


def default_dtype(device: torch.device) -> torch.dtype:
    """float32 on CUDA (fast), float64 on CPU."""
    return torch.float32 if device.type == "cuda" else torch.float64


def as_image_tensor(
    image_rgb,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """HxWx3 RGB -> float tensor on device."""
    if isinstance(image_rgb, torch.Tensor):
        tensor = image_rgb
    else:
        tensor = torch.as_tensor(np.asarray(image_rgb))
    return tensor.to(device=device, dtype=dtype)


def standardize_brightness(
    image: torch.Tensor, percentile: float = 90.0
) -> torch.Tensor:
    """Scale so the given intensity percentile maps to 255 (stainNorm utils)."""
    flat = image.reshape(-1).float()
    p = torch.quantile(flat, percentile / 100.0)
    if float(p) <= 0:
        return image.clamp(0, 255)
    return (image.float() * (255.0 / p)).clamp(0, 255).to(dtype=image.dtype)


def rgb_to_od(image_rgb: torch.Tensor) -> torch.Tensor:
    """OD = -log(I / 255), zeros replaced by 1."""
    image = image_rgb.clone()
    image = torch.where(image == 0, torch.ones_like(image), image)
    return -torch.log(image / 255.0)


def od_to_rgb(od: torch.Tensor) -> torch.Tensor:
    """RGB = 255 * exp(-OD) as uint8."""
    rgb = 255.0 * torch.exp(-od)
    return rgb.clamp(0, 255).to(torch.uint8)


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
    return matrix / torch.clamp(norms, min=eps)


def normalize_columns(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.norm(matrix, dim=0, keepdim=True)
    return matrix / torch.clamp(norms, min=eps)


def notwhite_mask_numpy(image_rgb_u8: np.ndarray, thresh: float = 0.8) -> np.ndarray:
    """LAB luminance mask on CPU (OpenCV), same as stainNorm_Vahadane."""
    if cv2 is None:
        raise ImportError("opencv-python is required for notwhite_mask")
    lab = cv2.cvtColor(np.asarray(image_rgb_u8, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float64) / 255.0
    return L < thresh


def dictionary_backend() -> str:
    """Which backend is used for Vahadane dictionary learning."""
    if _HAS_SPAMS:
        return "spams.trainDL"
    try:
        import sklearn  # noqa: F401

        return "sklearn.MiniBatchDictionaryLearning"
    except ImportError:
        return "snmf_fallback"


def order_stains_purple_first(dictionary: torch.Tensor) -> torch.Tensor:
    """Order stain rows so the more purple/magenta stain comes first.

    Converts OD stain vectors to RGB appearance (255*exp(-OD)). Purple/magenta
    has high R+B and low G; green stains (common MGG failure mode) score low.
    This replaces the H&E-only heuristic of sorting by red OD channel.
    """
    if dictionary.shape[0] < 2:
        return dictionary
    colors = torch.exp(-dictionary)  # relative RGB appearance
    purple_score = colors[:, 0] + colors[:, 2] - colors[:, 1]
    if purple_score[0] >= purple_score[1]:
        return dictionary
    return dictionary.flip(0)


def get_concentrations(
    od_flat: torch.Tensor,
    stain_matrix: torch.Tensor,
    lamda: float = 0.01,
) -> torch.Tensor:
    """Estimate concentrations (N x 2).

    Prefer SPAMS lasso (stainNorm reference). Fallback: clipped least squares on device.
    """
    device = od_flat.device
    dtype = od_flat.dtype

    if _HAS_SPAMS:
        od_3n = np.asfortranarray(
            od_flat.detach().cpu().numpy().T, dtype=np.float64
        )  # (3, N)
        stain_3k = np.asfortranarray(
            stain_matrix.detach().cpu().numpy().T, dtype=np.float64
        )  # (3, 2)
        coef = spams.lasso(
            od_3n,
            D=stain_3k,
            mode=2,
            lambda1=lamda,
            pos=True,
        )
        concentrations = np.asarray(coef.toarray().T, dtype=np.float64)  # (N, 2)
        return torch.as_tensor(concentrations, device=device, dtype=dtype)

    solution = torch.linalg.lstsq(stain_matrix.T, od_flat.T, rcond=None).solution
    return torch.clamp(solution.T, min=0.0)


def train_stain_dictionary(
    od_tissue: torch.Tensor,
    *,
    n_stains: int = 2,
    lambda1: float = 0.1,
    max_iter: int = 100,
    random_state: int = 0,
) -> torch.Tensor:
    """Learn stain dictionary (K, 3).

    Prefer SPAMS ``trainDL``; else sklearn positive dictionary learning;
    else multiplicative SNMF on device.
    ``od_tissue`` is (N_tissue, 3).
    """
    device = od_tissue.device
    dtype = od_tissue.dtype

    if _HAS_SPAMS:
        od_np = np.asfortranarray(
            od_tissue.detach().cpu().numpy().T, dtype=np.float64
        )  # (3, N)
        dictionary = spams.trainDL(
            od_np,
            K=n_stains,
            lambda1=lambda1,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False,
        ).T  # (K, 3)
        return torch.as_tensor(dictionary, device=device, dtype=dtype)

    # sklearn is closer to SPAMS trainDL than plain SNMF (available on Windows).
    try:
        from sklearn.decomposition import MiniBatchDictionaryLearning

        x = od_tissue.detach().cpu().numpy().astype(np.float64)
        # Cap samples for speed on large images.
        rng = np.random.default_rng(random_state)
        if x.shape[0] > 20000:
            idx = rng.choice(x.shape[0], size=20000, replace=False)
            x = x[idx]
        dl = MiniBatchDictionaryLearning(
            n_components=n_stains,
            alpha=lambda1,
            max_iter=max(max_iter, 200),
            fit_algorithm="cd",
            transform_algorithm="lasso_lars",
            positive_code=True,
            positive_dict=True,
            random_state=random_state,
            batch_size=min(256, max(32, x.shape[0] // 10)),
        )
        dl.fit(x)
        dictionary = np.maximum(dl.components_, 0.0)  # (K, 3)
        return torch.as_tensor(dictionary, device=device, dtype=dtype)
    except Exception:
        pass

    return snmf_dictionary(
        od_tissue.T,
        n_stains=n_stains,
        lambda1=lambda1,
        max_iter=max_iter,
        random_state=random_state,
    )


def snmf_dictionary(
    od_matrix: torch.Tensor,
    *,
    n_stains: int = 2,
    lambda1: float = 0.1,
    max_iter: int = 100,
    random_state: int = 0,
) -> torch.Tensor:
    """Sparse NMF fallback. od_matrix (3, N) -> dictionary (K, 3)."""
    n_channels, n_pixels = od_matrix.shape
    k = n_stains
    device = od_matrix.device
    dtype = od_matrix.dtype

    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)

    w = (
        torch.rand((n_channels, k), device=device, dtype=dtype, generator=generator)
        + 1e-3
    )
    h = (
        torch.rand((k, n_pixels), device=device, dtype=dtype, generator=generator)
        + 1e-3
    )
    w = normalize_columns(w)

    eps = 1e-8
    for _ in range(max_iter):
        numerator_h = w.T @ od_matrix
        denominator_h = (w.T @ w) @ h + lambda1 + eps
        h = torch.clamp(h * (numerator_h / denominator_h), min=0.0)

        numerator_w = od_matrix @ h.T
        denominator_w = w @ (h @ h.T) + eps
        w = torch.clamp(w * (numerator_w / denominator_w), min=0.0)
        w = normalize_columns(w)

    return w.T  # (K, 3)
