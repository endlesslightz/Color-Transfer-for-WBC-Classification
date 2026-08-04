"""Iterative Color Transfer Methods"""

import scipy
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from methods.metrics import rmse, psnr, ssim
from methods.CTQM import ctqm
import networkx as nx
import cv2
from fcmeans import FCM
from copy import deepcopy

def iterative_distribution_transfer(target, ref, bins=255, n_iter=4):
    """Iterative Distribution Transfer

    Citation
    --------
    @article{pitie2007automated,
      title={Automated colour grading using colour distribution transfer},
      author={Piti{\\'e}, Fran{\\c{c}}ois and Kokaram, Anil C and Dahyot, Rozenn},
      journal={Computer Vision and Image Understanding},
      volume={107},
      number={1-2},
      pages={123--137},
      year={2007},
      publisher={Elsevier}
    }
    """

    shape = target.shape
    n_dims = shape[-1]

    target = target.reshape(-1, 3)
    reference = ref.reshape(-1, 3)
    for _ in range(n_iter):
        r = scipy.stats.special_ortho_group.rvs(n_dims)

        d0r = r @ target.T
        d1r = r @ reference.T
        d_r = np.empty_like(target.T)

        for j in range(n_dims):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(float)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(float)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        target = np.linalg.solve(r, (d_r - d0r)).T + target

    output = target.reshape(shape)

    return output


def _regrain(img_arr_in, img_arr_col, nbits=(4, 16, 32, 64, 64, 64), level=0):
    h, w, _ = img_arr_in.shape
    h2 = (h + 1) // 2
    w2 = (w + 1) // 2

    if len(nbits) > 1 and h2 > 20 and w2 > 20:
        resize_arr_in = resize(img_arr_in, (h2, w2))
        resize_arr_col = resize(img_arr_col, (h2, w2))
        resize_arr_out = _regrain(resize_arr_in, resize_arr_col, nbits[1:], level + 1)
        img_arr_out = resize(resize_arr_out, (h, w))
    else:
        img_arr_out = img_arr_in

    img_arr_out = _solve(img_arr_out, img_arr_in, img_arr_col, nbits[0], level)

    return img_arr_out


def _solve(img_arr_out,
           img_arr_in,
           img_arr_col,
           nbit,
           level,
           eps=1e-6):
    width, height, c = img_arr_in.shape
    first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
    first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

    delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
    delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
    delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

    psi = 256 * delta / 5
    psi[psi > 1] = 1
    phi = 30 * 2**(-level) / (1 + 10 * delta)

    phi1 = (last_pad_1(phi) + phi) / 2
    phi2 = (last_pad_0(phi) + phi) / 2
    phi3 = (first_pad_1(phi) + phi) / 2
    phi4 = (first_pad_0(phi) + phi) / 2

    rho = 1 / 5.0
    for i in range(nbit):
        den = psi + phi1 + phi2 + phi3 + phi4
        num = (np.tile(psi, [1, 1, c]) * img_arr_col +
               np.tile(phi1, [1, 1, c]) * (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in) +
               np.tile(phi2, [1, 1, c]) * (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in) +
               np.tile(phi3, [1, 1, c]) * (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in) +
               np.tile(phi4, [1, 1, c]) * (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in))
        img_arr_out = (num / np.tile(den + eps, [1, 1, c]) * (1 - rho) + rho * img_arr_out)

    return img_arr_out


def automated_color_grading(target, reference, device=None):
    """Automated Colour Grading using Colour Distribution Transfer

    Citation
    --------
    @article{pitie2007automated,
      title={Automated colour grading using colour distribution transfer},
      author={Piti{\\'e}, Fran{\\c{c}}ois and Kokaram, Anil C and Dahyot, Rozenn},
      journal={Computer Vision and Image Understanding},
      volume={107},
      number={1-2},
      pages={123--137},
      year={2007},
      publisher={Elsevier}
    }
    """

    # Allocate on GPU to measure memory footprint
    if device is not None and device.type == "cuda":
        _src_gpu = torch.from_numpy(target.astype(np.float32)).to(device)
        _ref_gpu = torch.from_numpy(reference.astype(np.float32)).to(device)

    output = iterative_distribution_transfer(target, reference)
    output = _regrain(target, output)

    ref_height, ref_width = target.shape[:2]   
    reference  = cv2.resize(reference, (ref_width, ref_height))

    # Free GPU tensors
    if device is not None and device.type == "cuda":
        del _src_gpu, _ref_gpu

    out = (output * 255).astype(np.uint8)
    ref = (reference * 255).astype(np.uint8)

    # RMSE = rmse(out, ref)
    # PSNR = psnr(out, ref)
    # SSIM = ssim(out, ref)
    # CTQM = ctqm(out, ref)
    return output, 0, 0, 0, 0


def get_colors(img):
        ref_height, ref_width = img.shape[:2] 
        return img.reshape(ref_width * ref_height , 1, 3)


def set_colors(img):
        ref_height, ref_width = img.shape[:2] 
        img = img.reshape(ref_width, ref_height, 3).astype(np.float32)

def fuzzy(target, reference):
   
 # Record the start time for performance measurement
    # Extract colors from the source and reference images
    src_color = get_colors(target)
    ref_color = get_colors(reference)
    out_img = deepcopy(target)

    # [1] Extract parameters needed for the algorithm
    src_pix_num = src_color.shape[0]
    ref_pix_num = ref_color.shape[0]
    dim = src_color.shape[2]
    clusters = 3
    max_iter = 100
    fuzzier =2.0
    term_error = 1e-04

    # [2] Convert colors from RGB to Lab color space
    src_color = rgb2lab(src_color)
    ref_color = rgb2lab(ref_color)


    # [3] reshaping to (num_pixels, 3) because input is of size (num_pixels, 1, 3)
    src_reshape = src_color.reshape(src_pix_num, dim)
    ref_reshape = ref_color.reshape(ref_pix_num, dim)

    # [4] Apply Fuzzy C-Means clustering to both source and reference colors
    fcm_src = FCM(n_clusters=clusters, max_iter=max_iter, m=fuzzier, error=term_error)
    fcm_src.fit(src_reshape)

    fcm_ref = FCM(n_clusters=clusters, max_iter=max_iter, m=fuzzier, error=term_error)
    fcm_ref.fit(ref_reshape)


    # [5] Calculate cluster directions
    membership_s = fcm_src.u
    norm_factor_s = np.sum(membership_s, axis=0)
    centers_s = fcm_src.centers

    std_s = np.zeros((clusters, 3))
    weights_s = np.zeros(clusters)
    for c in range(clusters):
        sig_l = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,0:1] - centers_s[c][0], 2) / norm_factor_s[c]))
        sig_a = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,1:2] - centers_s[c][1], 2) / norm_factor_s[c]))
        sig_b = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,2:3] - centers_s[c][2], 2) / norm_factor_s[c]))
        std_s[c] = np.array([sig_l, sig_a, sig_b])
        weights_s[c] = (1/3)*sig_l + (1/3)*sig_a + (1/3)*sig_b

    membership_r = fcm_ref.u
    norm_factor_r = np.sum(membership_r, axis=0)
    centers_r = fcm_ref.centers
    std_r = np.zeros((clusters, 3))
    weights_r = np.zeros(clusters)
    for c in range(clusters):
        sig_l = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,0:1] - centers_r[c][0], 2) / norm_factor_r[c]))
        sig_a = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,1:2] - centers_r[c][1], 2) / norm_factor_r[c]))
        sig_b = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,2:3] - centers_r[c][2], 2) / norm_factor_r[c]))
        std_r[c] = np.array([sig_l, sig_a, sig_b])
        weights_r[c] = (1/3)*sig_l + (1/3)*sig_a + (1/3)*sig_b

    # [6] Calculate cluster directions by bipartite matching
    mapping = np.arange(clusters)
    B = nx.Graph()
    B.add_nodes_from(np.arange(clusters), bipartite=0)
    B.add_nodes_from(np.arange(2*clusters), bipartite=1)

    for ks, ws in enumerate(weights_s):
        for kr, wr in enumerate(weights_r):
            B.add_edge(ks, kr+clusters, weight=np.linalg.norm(ws-wr))

    my_matching = nx.bipartite.matching.minimum_weight_full_matching(B, np.arange(clusters), "weight")

    for i in range(clusters):
        mapping[i] = my_matching[i] - clusters

    # [7] Apply Reinhard's Color Transfer per cluster combination
    lab_new = np.zeros((dim, src_pix_num))

    for c in range(clusters):
        l_c = (std_r[mapping[c]][0]/std_s[c][0]) * (src_reshape[:,0:1] - centers_s[c][0]) + centers_r[mapping[c]][0]
        a_c = (std_r[mapping[c]][1]/std_s[c][1]) * (src_reshape[:,1:2] - centers_s[c][1]) + centers_r[mapping[c]][1]
        b_c = (std_r[mapping[c]][2]/std_s[c][2]) * (src_reshape[:,2:3] - centers_s[c][2]) + centers_r[mapping[c]][2]

        lab_new[0] += np.sum(l_c * membership_s[:,c:c+1], axis=1)
        lab_new[1] += np.sum(a_c * membership_s[:,c:c+1], axis=1)
        lab_new[2] += np.sum(b_c * membership_s[:,c:c+1], axis=1)

    lab_new = lab_new.T.reshape((src_pix_num, dim))

    # [8] Convert the resulting Lab colors back to RGB
    lab_new = lab_new.reshape(src_pix_num, 1, dim)
    lab_new = lab2rgb(lab_new)
    lab_new = np.clip(lab_new, 0.0, 1.0)

    # [9] Set the new colors to the output image
    out_img = set_colors(lab_new)

    # Prepare and return the output with performance metrics
    output =  out_img

    # RMSE = rmse(out, ref)
    # PSNR = psnr(out, ref)
    # SSIM = ssim(out, ref)
    return output