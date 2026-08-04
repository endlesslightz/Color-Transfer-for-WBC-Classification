"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

# from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer, ColorTransferEvaluation

import matplotlib.pyplot as plt
import cv2
import os
import sys
import gc
import time
import torch
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

# ------------------------------------------------------------------------------------------------------------------
import tracemalloc

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Logging setup: duplicate all output to log_deephist.txt ---
        log_path = os.path.join(base_dir, 'log_deephist.txt')
        sys.stdout = TeeLogger(log_path)

        folder_path = os.path.join(base_dir, 'images', 'eksperimen')
        save_folder = os.path.join(base_dir, 'images', 'hasil')
        ref_path = os.path.join(base_dir, 'images', 'Referensi.jpg')

        if not os.path.exists(folder_path):
                raise ValueError(f"Folder tidak ditemukan: {folder_path}")
        os.makedirs(save_folder, exist_ok=True)
        ref = Image(file_path=ref_path)  
        
        # --- Device & GPU memory setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = device.type == "cuda"
        print(f"[DeepCT] Device: {device}\n")

        for algo in ["PCC"]:
            print("=" * 60)
            print(f"Running: {algo}")
            print("=" * 60)

            algo_save_folder = os.path.join(base_dir, 'images', algo.lower())
            os.makedirs(algo_save_folder, exist_ok=True)
            
            tracemalloc.start()
            tracemalloc.clear_traces()

            times_s = []

            for file_name in os.listdir(folder_path):
                fpath = os.path.join(folder_path, file_name)
                # Periksa apakah file adalah citra
                if os.path.isfile(fpath) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image = Image(file_path=fpath)
                    
                    save_path = os.path.join(algo_save_folder, file_name)

                    # --- Timing start ---
                    t0 = time.perf_counter()

                    ct = ColorTransfer(image, ref, algo)
                    out = ct.apply()

                    if use_cuda:
                        torch.cuda.synchronize()

                    elapsed = time.perf_counter() - t0
                    times_s.append(elapsed)
                    # --- Timing end ---

                    # Track peak GPU memory
                    # removed GPU tracking block

                    out["object"].write(out_path=save_path)

                    # Pembersihan memory untuk mencegah OOM
                    del ct, out, image
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            current_ram, peak_ram_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_ram_mib = peak_ram_bytes / (1024 ** 2)

            # --- Performance summary ---
            avg_s = sum(times_s) / len(times_s) if times_s else 0.0
            print(f"\n[{algo}] --- Performance ---")
            print(f"[{algo}] Average Processing Time   : {avg_s:.3f} s/image")
            print(f"[{algo}] RAM Memory Consumption    : Peak Delta = {peak_ram_mib:.1f} MiB")
            print()

        print("Done")
