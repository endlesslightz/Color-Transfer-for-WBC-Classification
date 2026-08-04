"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import cv2

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Root Mean Square Error (RMSE)
# 
#
# Source: 
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class RMSE:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = (args[2].get_raw() * 255).astype(np.uint8) if hasattr(args[2], 'get_raw') else args[2]
        ref = (args[1].get_raw() * 255).astype(np.uint8) if hasattr(args[1], 'get_raw') else args[1]
        if ref.shape != src.shape:
            ref = cv2.resize(ref, (src.shape[1], src.shape[0]))
        valmse = np.mean((ref - src) ** 2)
        return np.sqrt(valmse)
    