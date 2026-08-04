import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity
import cv2

# def rmse(image1, image2):
#     if image1.shape != image2.shape:
#         raise ValueError("Kedua citra harus memiliki dimensi yang sama")
#     blue_img1, green_img1, red_img1 = cv2.split(image1)
#     blue_img2, green_img2, red_img2 = cv2.split(image2)
#     
#     # Menghitung perbedaan antara kedua citra
#     diff1 = blue_img1 - blue_img2
#     diff2 = green_img1 - green_img2
#     diff3 = red_img1 - red_img2
#     
#     # Menghitung RMSE untuk keseluruhan citra
#     mse = (np.mean(np.square(diff1))+np.mean(np.square(diff2))+np.mean(np.square(diff3)))/3
#     rmse = np.sqrt(mse)
#     return rmse

def rmse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Kedua citra harus memiliki dimensi yang sama")
    # Keep uint8 arithmetic to match [ok]cagan/metrics.py evaluation protocol.
    valmse = np.mean((image1 - image2) ** 2)
    return np.sqrt(valmse)

def psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Kedua citra harus memiliki dimensi yang sama")
    # Keep uint8 arithmetic to match [ok]cagan/metrics.py evaluation protocol.
    valmse = np.mean((image1 - image2) ** 2)
    if valmse == 0:
        return float('inf')  # PSNR tidak terhingga jika tidak ada perbedaan
    max_pixel = 255.0
    psnr2 = 10 * np.log10((max_pixel ** 2) / valmse)
    return psnr2

def ssim(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Kedua citra harus memiliki dimensi yang sama")
    blue_img1, green_img1, red_img1 = cv2.split(image1)
    blue_img2, green_img2, red_img2 = cv2.split(image2)
    skor1 = structural_similarity(blue_img1, blue_img2, data_range=255, 
                               gaussian_weights=True, sigma=1.5, 
                               use_sample_covariance=False)
    skor2 = structural_similarity(green_img1, green_img2, data_range=255, 
                               gaussian_weights=True, sigma=1.5, 
                               use_sample_covariance=False)
    skor3 = structural_similarity(red_img1, red_img2, data_range=255, 
                               gaussian_weights=True, sigma=1.5, 
                               use_sample_covariance=False)
    ssim = (skor1+skor2+skor3)/3
    return ssim
