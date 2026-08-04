import argparse
import sys

import numpy as np
import os
from PIL import Image
import cv2

class PCC:
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud"],
        "ref": ["Image", "Mesh", "PointCloud"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Returns basic information of the corresponding publication.
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "PCC",
            "title": "Kotera",
            "year": 2005,
            "abstract": "Each image has its own color content that greatly influences the perception of human "
                        "observer. Recently, color transfer among different images has been under investigation. In "
                        "this paper, after a brief review on the few efficient works performed in the field, a novel "
                        "fuzzy clustering based color transfer method is proposed. The proposed method accomplishes "
                        "the transformation based on a set of corresponding fuzzy clustering algorithm-selected "
                        "regions in images along with membership degree factors. Results show the presented algorithm "
                        "is highly automatically and more effective.",
            "types": ["Image"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        import time
        from copy import deepcopy
        
        start_time = time.time()
        
        # src and ref are ColorTransferLib.ImageProcessing.Image objects
        out_img = deepcopy(src)

        # src.get_raw() returns float32 array in range [0, 1]
        content = src.get_raw() * 255.0
        reference = ref.get_raw() * 255.0

        # Convert HxWxC image to a (H*W)xC matrix.
        shape = content.shape
        assert len(shape) == 3
        content = content.reshape(-1, shape[-1]).astype(np.float32)
        reference = reference.reshape(-1, shape[-1]).astype(np.float32)

        mu_content = np.mean(content, axis=0)
        mu_reference = np.mean(reference, axis=0)

        cov_content = np.cov(content, rowvar=False)
        cov_reference = np.cov(reference, rowvar=False)

        eigval_content, eigvec_content = np.linalg.eig(cov_content)
        eigval_reference, eigvec_reference = np.linalg.eig(cov_reference)

        scaling = np.diag(np.sqrt(eigval_reference / eigval_content))
        transform = eigvec_reference.dot(scaling).dot(eigvec_content.T)
        result = (content - mu_content).dot(transform.T) + mu_reference
        # Restore image dimensions.
        result = result.reshape(shape).clip(0, 255).round().astype(np.uint8)

        # Set the raw output to the new image object
        out_img.set_raw(result, normalized=False)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
