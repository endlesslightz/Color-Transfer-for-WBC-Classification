import os
import sys
import time
import cv2
import torch
import numpy as np
from skimage.util import img_as_float
from PIL import Image
import pandas as pd


class TeeLogger:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
from methods.linear import color_transfer_between_images as ct
from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
from methods.linear import monge_kantorovitch_color_transfer as mkct
from methods.iterative import automated_color_grading as acg

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import tracemalloc

# ------------------------------------------------------------------------------------------------------------------
# Color Transfer Runner with Performance Tracking
# ------------------------------------------------------------------------------------------------------------------
def run_method(method_name, method_func, folder_path, save_folder, reference, device):
    """Run a single color transfer method on all images with timing and GPU memory tracking.
    
    Parameters
    ----------
    method_name : str
        Display name for the method.
    method_func : callable
        Function(src, ref, device=device) -> (output, RMSE, PSNR, SSIM, CTQM).
    folder_path : str
        Input images directory.
    save_folder : str
        Output directory for this method.
    reference : ndarray
        Reference image (float [0,1]).
    device : torch.device
        CUDA or CPU device.

    Returns
    -------
    list of result rows: [filename, algorithm, RMSE, PSNR, SSIM, CTQM]
    """
    os.makedirs(save_folder, exist_ok=True)

    tracemalloc.start()
    tracemalloc.clear_traces()

    times_s = []
    hasil = []
    idx = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not (os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
            continue

        image = img_as_float(Image.open(file_path))
        if image is None:
            print(f"Gagal membaca file: {file_name}")
            continue

        idx += 1

        # --- Timing start ---
        t0 = time.perf_counter()

        result_img, rmse_val, psnr_val, ssim_val, ctqm_val = method_func(image, reference, device=device)

        elapsed = time.perf_counter() - t0
        times_s.append(elapsed)
        # --- Timing end ---

        # Track peak GPU memory
        # removed GPU tracking block

        # Save output
        save_path = os.path.join(save_folder, file_name)
        result_clipped = np.clip(result_img, 0, 1)
        out_uint8 = cv2.normalize(result_clipped, dst=None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        out_uint8 = cv2.cvtColor(out_uint8, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, out_uint8)

        hasil.append([file_name, method_name, rmse_val, psnr_val, ssim_val, ctqm_val])
        # print(f"  [{method_name}] {idx:>4d} {file_name} | Time: {elapsed:.3f}s")

    current_ram, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mib = peak_ram_bytes / (1024 ** 2)

    # --- Performance summary ---
    avg_s = sum(times_s) / len(times_s) if times_s else 0.0
    print(f"\n[{method_name}] --- Performance ---")
    # print(f"[{method_name}] Images processed          : {len(times_s)}")
    print(f"[{method_name}] Average Processing Time   : {avg_s:.3f} s/image")
    print(f"[{method_name}] RAM Memory Consumption    : Peak Delta = {peak_ram_mib:.1f} MiB")
    print()

    return hasil


# ------------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Logging setup: duplicate all output to log_ct.txt ---
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_ct.txt')
    sys.stdout = TeeLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, 'images', 'bahan') + os.sep
    reference = img_as_float(Image.open(os.path.join(script_dir, 'images', 'Referensi.jpg')).convert('RGB'))

    all_results = []

    # ---- Reinhard (Color Transfer between Images) ----
    print("=" * 60)
    print("Running: Reinhard (Color Transfer between Images)")
    print("=" * 60)
    results = run_method(
        method_name="Reinhard",
        method_func=ct,
        folder_path=folder,
        save_folder=os.path.join(script_dir, 'images', 'reinhard') + os.sep,
        reference=reference,
        device=device,
    )
    all_results.extend(results)

    # ---- Xiao/Kotera (Correlated Color Space) ----
    print("=" * 60)
    print("Running: Xiao (Color Transfer in Correlated Color Space)")
    print("=" * 60)
    results = run_method(
        method_name="Xiao",
        method_func=ct_ccs,
        folder_path=folder,
        save_folder=os.path.join(script_dir, 'images', 'xiao') + os.sep,
        reference=reference,
        device=device,
    )
    all_results.extend(results)

    # ---- Monge-Kantorovitch ----
    print("=" * 60)
    print("Running: Monge-Kantorovitch")
    print("=" * 60)
    results = run_method(
        method_name="MK",
        method_func=mkct,
        folder_path=folder,
        save_folder=os.path.join(script_dir, 'images', 'mk') + os.sep,
        reference=reference,
        device=device,
    )
    all_results.extend(results)

    # ---- Iterative Distribution Transfer (Automated Color Grading) ----
    print("=" * 60)
    print("Running: Iterative Distribution Transfer")
    print("=" * 60)
    results = run_method(
        method_name="IterDist",
        method_func=acg,
        folder_path=folder,
        save_folder=os.path.join(script_dir, 'images', 'iterdist') + os.sep,
        reference=reference,
        device=device,
    )
    all_results.extend(results)

    # ---- Save all results ----
    # columns = ['Nama', 'Algoritma', 'RMSE', 'PSNR', 'SSIM', 'CTQM']
    # df = pd.DataFrame(all_results, columns=columns)
    # out_excel = os.path.join(script_dir, 'output.xlsx')
    # df.to_excel(out_excel, index=False)
    # print(f"\nAll results saved to {out_excel} ({len(all_results)} rows)")
