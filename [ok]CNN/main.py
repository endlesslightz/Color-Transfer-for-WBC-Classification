import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

def calculate_rmse(imageA, imageB):
    return np.sqrt(np.mean((imageA - imageB) ** 2))

def calculate_psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if mse == 0:  # Avoid division by zero
        return float('inf')
    pixel_max = 255.0
    return 20 * log10(pixel_max / sqrt(mse))

def calculate_ssim(imageA, imageB):
    return ssim(imageA, imageB, multichannel=True)

def process_images(reference_path, input_folder):
    # Load the reference image
    reference_image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    if reference_image is None:
        raise FileNotFoundError(f"Reference image '{reference_path}' not found.")

    # Convert to grayscale if needed
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # Initialize metrics
    total_rmse, total_psnr, total_ssim = 0, 0, 0
    file_count = 0

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not os.path.isfile(file_path):
            continue

        # Load input image
        input_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if input_image is None:
            print(f"Warning: Could not load image '{file_path}'. Skipping...")
            continue

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Ensure both images are the same size
        if reference_image.shape != input_image.shape:
            input_image = cv2.resize(input_image, (reference_image.shape[1], reference_image.shape[0]))

        # Compute metrics
        rmse = calculate_rmse(reference_image, input_image)
        psnr = calculate_psnr(reference_image, input_image)
        ssim_value = calculate_ssim(reference_image, input_image)

        total_rmse += rmse
        total_psnr += psnr
        total_ssim += ssim_value
        file_count += 1

    # Compute average metrics
    if file_count == 0:
        raise ValueError("No valid images found in the input folder.")

    avg_rmse = total_rmse / file_count
    avg_psnr = total_psnr / file_count
    avg_ssim = total_ssim / file_count

    return avg_rmse, avg_psnr, avg_ssim

if __name__ == "__main__":
    reference_path = "reference.jpg"  # Path to the reference image
    input_folder = "input_images"     # Folder containing input images

    try:
        avg_rmse, avg_psnr, avg_ssim = process_images(reference_path, input_folder)
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    except Exception as e:
        print(f"Error: {e}")
