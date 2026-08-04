import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms


MODEL_TO_PATHS: Dict[str, Tuple[Path, Path]] = {
    "bestBaseline": (Path("checkpoint") / "bestBaseline.pt", Path("data") / "wbc" / "Val"),
    "bestCD": (Path("checkpoint") / "bestCD.pt", Path("data") / "colordis" / "Val"),
    "bestDeep": (Path("checkpoint") / "bestDeep.pt", Path("data") / "deep" / "Val"),
    "bestKotera": (Path("checkpoint") / "bestKotera.pt", Path("data") / "kotera" / "Val"),
    "bestMK": (Path("checkpoint") / "bestMK.pt", Path("data") / "mk" / "Val"),
    "bestXiao": (Path("checkpoint") / "bestXiao.pt", Path("data") / "xiao" / "Val"),
    "bestReinhard": (Path("checkpoint") / "bestReinhard.pt", Path("data") / "reinhard" / "Val"),
}


CKPT_STEM_TO_VAL_SUBDIR: Dict[str, str] = {
    "bestBaseline": "wbc",
    "bestCD": "colordis",
    "bestDeep": "deep",
    "bestKotera": "kotera",
    "bestMK": "mk",
    "bestXiao": "xiao",
    "bestReinhard": "reinhard",
}


class Modelku(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook data is empty. Check target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy(), class_idx, probs


class GradCAMResult:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path,
        val_dir: Path,
        image_path: Path,
        classes: List[str],
        pred_label: str,
        pred_conf: float,
        true_label: str,
        image_np: np.ndarray,
        heatmap_rgb: np.ndarray,
        overlay: np.ndarray,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.val_dir = val_dir
        self.image_path = image_path
        self.classes = classes
        self.pred_label = pred_label
        self.pred_conf = pred_conf
        self.true_label = true_label
        self.image_np = image_np
        self.heatmap_rgb = heatmap_rgb
        self.overlay = overlay


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def collect_images(val_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in val_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


def choose_image_interactively(images: List[Path], max_show: int = 120) -> Path:
    if not images:
        raise ValueError("Tidak ada gambar pada folder Val yang dipilih.")

    print(f"Jumlah gambar ditemukan: {len(images)}")
    print("Pilih 1 gambar (input nomor index).")

    show_count = min(len(images), max_show)
    for idx in range(show_count):
        print(f"[{idx}] {images[idx]}")

    if len(images) > max_show:
        print(f"... hanya menampilkan {max_show} pertama. Gunakan index 0..{len(images)-1}.")

    while True:
        raw = input("Masukkan index gambar: ").strip()
        if not raw.isdigit():
            print("Input harus angka.")
            continue
        idx = int(raw)
        if idx < 0 or idx >= len(images):
            print(f"Index di luar jangkauan (0..{len(images)-1}).")
            continue
        return images[idx]


def get_target_layer(model: Modelku, layer_name: str) -> nn.Module:
    current: nn.Module = model
    for part in layer_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            if not hasattr(current, part):
                raise AttributeError(f"Layer '{layer_name}' tidak ditemukan pada model.")
            current = getattr(current, part)
    return current


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint_state(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    raw = torch.load(ckpt_path, map_location=device)

    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            state = raw["model_state_dict"]
        else:
            state = raw
    else:
        raise ValueError("Format checkpoint tidak didukung.")

    return strip_module_prefix(state)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(cam)[..., :3]
    overlay = alpha * heatmap_rgb + (1.0 - alpha) * image_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return heatmap_rgb, overlay


def run_gradcam_for_model(
    model_choice: str,
    checkpoint_arg: Optional[str],
    val_dir_arg: Optional[str],
    image_path: Path,
    layer_name: str,
    alpha: float,
    image_size: int,
    device: torch.device,
) -> GradCAMResult:
    model_name, ckpt_path, val_dir = resolve_checkpoint_and_val_dir(
        model_choice=model_choice,
        checkpoint_arg=checkpoint_arg,
        val_dir_arg=val_dir_arg,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Folder validasi tidak ditemukan: {val_dir}")

    dataset = datasets.ImageFolder(root=val_dir)
    classes = dataset.classes
    num_classes = len(classes)

    model = Modelku(num_classes=num_classes).to(device)
    state = load_checkpoint_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] {model_name} missing keys: {len(missing)}")
    if unexpected:
        print(f"[Info] {model_name} unexpected keys: {len(unexpected)}")

    transform = get_transform(image_size)
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    image_np = np.array(pil_image.resize((image_size, image_size))).astype(np.float32) / 255.0

    target_layer = get_target_layer(model, layer_name)
    cam_generator = GradCAM(model, target_layer)

    try:
        cam, pred_idx, probs = cam_generator.generate(input_tensor)
    finally:
        cam_generator.remove_hooks()

    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])
    true_label = image_path.parent.name
    heatmap_rgb, overlay = overlay_heatmap(image_np, cam, alpha=alpha)

    return GradCAMResult(
        model_name=model_name,
        checkpoint_path=ckpt_path,
        val_dir=val_dir,
        image_path=image_path,
        classes=classes,
        pred_label=pred_label,
        pred_conf=pred_conf,
        true_label=true_label,
        image_np=image_np,
        heatmap_rgb=heatmap_rgb,
        overlay=overlay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM untuk 1 gambar pilihan dari folder Val.")
    parser.add_argument(
        "--model-choice",
        type=str,
        default="bestBaseline",
        choices=sorted(MODEL_TO_PATHS.keys()),
        help="Nama model preset (menentukan checkpoint dan val dir default).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override path checkpoint .pt",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Override folder validasi (Val)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path gambar spesifik. Jika tidak diisi, akan pilih manual dari daftar Val.",
    )
    parser.add_argument(
        "--compare-image",
        type=str,
        default=None,
        help="Path gambar untuk model pembanding saat memakai --compare-model.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        # default="model.layer4.2.conv3",
        default="model.layer3.5.conv3",
        help="Layer target untuk Grad-CAM.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Kekuatan heatmap pada overlay (0..1).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Ukuran resolusi input untuk inferensi dan visualisasi Grad-CAM.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path output gambar visualisasi. Default: outputs/gradcam_<model>_<nama_gambar>.png",
    )
    parser.add_argument(
        "--compare-model",
        type=str,
        default=None,
        choices=sorted(MODEL_TO_PATHS.keys()),
        help="Jika diisi, tampilkan perbandingan Grad-CAM antara model utama dan model ini.",
    )
    return parser.parse_args()


def resolve_checkpoint_and_val_dir(model_choice: str, checkpoint_arg: Optional[str], val_dir_arg: Optional[str]) -> Tuple[str, Path, Path]:
    if model_choice in MODEL_TO_PATHS:
        preset_name = model_choice
        default_ckpt, default_val = MODEL_TO_PATHS[preset_name]
        ckpt_path = Path(checkpoint_arg) if checkpoint_arg else default_ckpt
        val_dir = Path(val_dir_arg) if val_dir_arg else default_val
        return preset_name, ckpt_path, val_dir

    # Fallback: user passed checkpoint path in --model-choice.
    ckpt_path = Path(model_choice)
    if checkpoint_arg:
        ckpt_path = Path(checkpoint_arg)

    preset_name = ckpt_path.stem
    if val_dir_arg:
        val_dir = Path(val_dir_arg)
    else:
        subdir = CKPT_STEM_TO_VAL_SUBDIR.get(ckpt_path.stem)
        val_dir = Path("data") / subdir / "Val" if subdir else Path("data") / "wbc" / "Val"

    return preset_name, ckpt_path, val_dir


def main() -> None:
    args = parse_args()

    device = get_device()

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    else:
        _, _, val_dir = resolve_checkpoint_and_val_dir(
            model_choice=args.model_choice,
            checkpoint_arg=args.checkpoint,
            val_dir_arg=args.val_dir,
        )
        if not val_dir.exists():
            raise FileNotFoundError(f"Folder validasi tidak ditemukan: {val_dir}")
        image_list = collect_images(val_dir)
        image_path = choose_image_interactively(image_list)

    compare_image_path = image_path
    if args.compare_image:
        compare_image_path = Path(args.compare_image)
        if not compare_image_path.exists():
            raise FileNotFoundError(f"Gambar pembanding tidak ditemukan: {compare_image_path}")

    primary = run_gradcam_for_model(
        model_choice=args.model_choice,
        checkpoint_arg=args.checkpoint,
        val_dir_arg=args.val_dir,
        image_path=image_path,
        layer_name=args.layer,
        alpha=args.alpha,
        image_size=args.img_size,
        device=device,
    )

    if args.compare_model:
        comparison = run_gradcam_for_model(
            model_choice=args.compare_model,
            checkpoint_arg=None,
            val_dir_arg=None,
            image_path=compare_image_path,
            layer_name=args.layer,
            alpha=args.alpha,
            image_size=args.img_size,
            device=device,
        )

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes[0, 0].imshow(primary.image_np)
        axes[0, 0].set_title(f"Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(primary.heatmap_rgb)
        axes[0, 1].set_title(f"Baseline Heatmap")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(primary.overlay)
        axes[0, 2].set_title(f"Baseline Overlay")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(comparison.image_np)
        axes[1, 0].set_title(f"Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(comparison.heatmap_rgb)
        axes[1, 1].set_title(f"Color Distribution Transfer Heatmap")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(comparison.overlay)
        axes[1, 2].set_title(f"Color Distribution Transfer Overlay")
        axes[1, 2].axis("off")

        plt.tight_layout()

        output_path = Path(args.output) if args.output else Path("outputs") / f"gradcam_compare_{primary.model_name}_vs_{comparison.model_name}_{image_path.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()

        print("=== Grad-CAM comparison selesai ===")
        print(f"Model 1   : {primary.model_name} | Checkpoint: {primary.checkpoint_path} | Val dir: {primary.val_dir}")
        print(f"Model 2   : {comparison.model_name} | Checkpoint: {comparison.checkpoint_path} | Val dir: {comparison.val_dir}")
        print(f"Image 1   : {image_path}")
        print(f"Image 2   : {compare_image_path}")
        print(f"Pred 1    : {primary.pred_label} ({primary.pred_conf*100:.2f}%)")
        print(f"Pred 2    : {comparison.pred_label} ({comparison.pred_conf*100:.2f}%)")
        print(f"Output    : {output_path}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(primary.image_np)
    axes[0].set_title(f"Original\nTrue: {primary.true_label}")
    axes[0].axis("off")

    axes[1].imshow(primary.heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(primary.overlay)
    axes[2].set_title(f"Overlay\nPred: {primary.pred_label} ({primary.pred_conf*100:.2f}%)")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = Path(args.output) if args.output else Path("outputs") / f"gradcam_{primary.model_name}_{image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("=== Grad-CAM selesai ===")
    print(f"Model     : {primary.model_name}")
    print(f"Checkpoint: {primary.checkpoint_path}")
    print(f"Val dir   : {primary.val_dir}")
    print(f"Image     : {image_path}")
    print(f"Prediksi  : {primary.pred_label} ({primary.pred_conf*100:.2f}%)")
    print(f"Output    : {output_path}")


if __name__ == "__main__":
    main()
