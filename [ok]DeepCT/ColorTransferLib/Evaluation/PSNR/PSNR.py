"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import numpy as np

from ColorTransferLib.ImageProcessing.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: -
#   Author: -
#   Published in: -
#   Year of Publication: -
#
# Abstract:
#   -

# Info:
#   Name: Peak Signal-to-Noise Ratio
#   Shortname: PSNR
#   Identifier: PSNR
#   Link: -
#   Range: [0, inf] with higher values indicates better quality
#
# Implementation Details:
#   if source and output are identical, the PSNR will be not considered for the evaluation
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PSNR:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = (args[2].get_raw() * 255).astype(np.uint8) if hasattr(args[2], 'get_raw') else args[2]
        ref = (args[1].get_raw() * 255).astype(np.uint8) if hasattr(args[1], 'get_raw') else args[1]
        if src.shape != ref.shape:
            ref = cv2.resize(ref, (src.shape[1], src.shape[0]))
        valmse = np.mean((src - ref) ** 2)
        if valmse == 0:
            return float('inf')  # PSNR tidak terhingga jika tidak ada perbedaan
        max_pixel = 255.0
        valpsnr = 10 * np.log10((max_pixel ** 2) / valmse)
        return valpsnr