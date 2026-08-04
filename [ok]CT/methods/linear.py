"""Linear Color Transfer Methods"""

import scipy
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from methods.metrics import rmse, psnr,ssim
from methods.CTQM import ctqm
import cv2


def color_transfer_between_images(src, ref, device=None):
    """Color Transfer between Images

    Citation
    --------
    @article{reinhard2001color,
      title={Color transfer between images},
      author={Reinhard, Erik and Adhikhmin, Michael and Gooch, Bruce and Shirley, Peter},
      journal={IEEE Computer graphics and applications},
      volume={21},
      number={5},
      pages={34--41},
      year={2001},
      publisher={IEEE}
    }
    """

    # Allocate on GPU to measure memory footprint
    if device is not None and device.type == "cuda":
        _src_gpu = torch.from_numpy(src.astype(np.float32)).to(device)
        _ref_gpu = torch.from_numpy(ref.astype(np.float32)).to(device)

    target = rgb2lab(src)
    reference = rgb2lab(ref)

    shape = target.shape
    ref_height, ref_width = target.shape[:2]   
    ref  = cv2.resize(ref, (ref_width, ref_height))
    target = target.reshape(-1, 3)
    reference = reference.reshape(-1, 3)

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_std = np.std(target, axis=0)
    reference_std = np.std(reference, axis=0)

    output = (target - target_mean) * reference_std / target_std + reference_mean

    output = lab2rgb(output.reshape(shape))

    # Free GPU tensors
    if device is not None and device.type == "cuda":
        del _src_gpu, _ref_gpu

    # diubah ke bentuk opencv 8 bit
    out = (output * 255).astype(np.uint8)
    ref_float = ref
    ref = (ref * 255).astype(np.uint8)

    # RMSE = rmse(out, ref)
    # PSNR = psnr(out, ref)
    # SSIM = ssim(out, ref)
    # CTQM = ctqm(src, ref_float, output)
    return output, 0, 0, 0, 0


def color_transfer_in_correlated_color_space(src, ref, device=None):
    """Color Transfer in Correlated Color Space

    Citation
    --------
    @inproceedings{xiao2006color,
      title={Color transfer in correlated color space},
      author={Xiao, Xuezhong and Ma, Lizhuang},
      booktitle={Proceedings of the 2006 ACM international conference on Virtual reality continuum and its applications},
      pages={305--309},
      year={2006}
    }
    """

    # Allocate on GPU to measure memory footprint
    if device is not None and device.type == "cuda":
        _src_gpu = torch.from_numpy(src.astype(np.float32)).to(device)
        _ref_gpu = torch.from_numpy(ref.astype(np.float32)).to(device)

    shape = src.shape

    target = src.reshape(-1, 3)
    reference = ref.reshape(-1, 3)
    ref_height, ref_width = src.shape[:2]   
    ref  = cv2.resize(ref, (ref_width, ref_height))

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_cov = np.cov(target.T)
    reference_cov = np.cov(reference.T)

    target_u, target_s, _ = np.linalg.svd(target_cov)
    reference_u, reference_s, _ = np.linalg.svd(reference_cov)

    target_rotation = target_u
    reference_rotation = np.linalg.inv(reference_u)

    target_scale = np.diag(1 / np.sqrt(target_s))
    reference_scale = np.diag(np.sqrt(reference_s))

    T = target_rotation @ target_scale @ reference_scale @ reference_rotation

    output = (target - target_mean) @ T.T + reference_mean

    # Free GPU tensors
    if device is not None and device.type == "cuda":
        del _src_gpu, _ref_gpu

    out = (output.reshape(shape) * 255).astype(np.uint8)
    ref_float = ref
    ref = (ref * 255).astype(np.uint8)

    # RMSE = rmse(out, ref)
    # PSNR = psnr(out, ref)
    # SSIM = ssim(out, ref)
    # CTQM = ctqm(src, ref_float, np.clip(output.reshape(shape), 0, 1))
    return output.reshape(shape), 0, 0, 0, 0


def Color_Transfer_CCS(source, target):
    rgb_s = np.reshape(source, (-1, 3)).T
    rgb_t = np.reshape(target, (-1, 3)).T

    mean_s = np.mean(rgb_s, axis=1)
    mean_t = np.mean(rgb_t, axis=1)

    cov_s = np.cov(rgb_s)
    cov_t = np.cov(rgb_t)

    U_s, A_s, _ = np.linalg.svd(cov_s)
    U_t, A_t, _ = np.linalg.svd(cov_t)

    rgbh_s = np.vstack((rgb_s, np.ones((1, rgb_s.shape[1]), dtype=np.float64)))

    T_t = np.eye(4, dtype=np.float64)
    T_t[:3, 3] = mean_t
    T_s = np.eye(4, dtype=np.float64)
    T_s[:3, 3] = -mean_s

    R_t = np.block([[U_t, np.zeros((3, 1), dtype=np.float64)], [np.zeros((1, 3), dtype=np.float64), 1]])
    R_s = np.block([[np.linalg.inv(U_s), np.zeros((3, 1), dtype=np.float64)], [np.zeros((1, 3), dtype=np.float64), 1]])

    A_t_sqrt = np.sqrt(A_t)
    S_t = np.diag(A_t_sqrt.tolist() + [1])
    A_s_inv_sqrt = np.power(A_s, -0.5)
    S_s = np.diag(A_s_inv_sqrt.tolist() + [1])

    rgbh_e = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(T_t, R_t), S_t), S_s), R_s), T_s).dot(rgbh_s.astype(np.float64))
    rgb_e = rgbh_e[:3, :]

    est_im = np.reshape(rgb_e.T, source.shape)

    return est_im


def monge_kantorovitch_color_transfer(src, ref, decomposition="MK", device=None):
    """The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer

    Citation
    --------
    @article{pitie2007linear,
      title={The linear monge-kantorovitch linear colour mapping for example-based colour transfer},
      author={Piti{\\'e}, Fran{\\c{c}}ois and Kokaram, Anil},
      year={2007},
      publisher={IET}
    }
    """

    # Allocate on GPU to measure memory footprint
    if device is not None and device.type == "cuda":
        _src_gpu = torch.from_numpy(src.astype(np.float32)).to(device)
        _ref_gpu = torch.from_numpy(ref.astype(np.float32)).to(device)

    shape = src.shape

    target = src.reshape(-1, 3)
    reference = ref.reshape(-1, 3)
    ref_height, ref_width = src.shape[:2]   
    ref  = cv2.resize(ref, (ref_width, ref_height))

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_cov = np.cov(target.T)
    reference_cov = np.cov(reference.T)

    if decomposition == "cholesky":
        A = np.linalg.cholesky(target_cov)
        B = np.linalg.cholesky(reference_cov)
        T = B @ np.linalg.inv(A)
    elif decomposition == "sqrt":
        A = scipy.linalg.sqrtm(target_cov)
        B = scipy.linalg.sqrtm(reference_cov)
        T = B @ np.linalg.inv(A)
    elif decomposition == "MK":
        A = scipy.linalg.sqrtm(target_cov)
        T = np.linalg.inv(A) @ scipy.linalg.sqrtm(A @ reference_cov @ A) @ np.linalg.inv(A)
    else:
        raise ValueError("Unknown decomposition, use either 'cholesky', 'sqrt', or 'MK'")

    output = (target - target_mean) @ T + reference_mean

    # Free GPU tensors
    if device is not None and device.type == "cuda":
        del _src_gpu, _ref_gpu

    out = (output.reshape(shape) * 255).astype(np.uint8)
    ref_float = ref
    ref = (ref * 255).astype(np.uint8)

    # RMSE = rmse(out, ref)
    # PSNR = psnr(out, ref)
    # SSIM = ssim(out, ref)
    # CTQM = ctqm(src, ref_float, np.clip(output.reshape(shape), 0, 1))
    return output.reshape(shape), 0, 0, 0, 0