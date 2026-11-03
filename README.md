# Color Transfer for Enhanced White Blood Cell Classification in Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: **"Color transfer for enhanced white blood cell classification in deep learning"**. 

## 📖 Overview
Inconsistencies in blood smear staining protocols (e.g., May-Grünwald-Giemsa, Wright, Giemsa) often cause "domain shifts" that degrade the performance of automated Hematology AI systems. This study comprehensively evaluates **nine color transfer algorithms** (both statistical and deep learning-based) as preprocessing steps to harmonize peripheral blood smear images. 

We empirically demonstrate that the **Monge-Kantorovich linear color mapping** optimally preserves leukocyte morphological integrity, restricts model spatial attention (Grad-CAM) to actual cellular structures, and significantly boosts CNN classification accuracy (up to 94.46% on ResNet-50) across both primary and unseen external datasets.

![Proposed Framework](framework_placeholder.png)
*(Note: You can upload your Fig. 1 to the repo and replace `framework_placeholder.png` with the actual path)*

## 🔬 Evaluated Color Transfer Algorithms
1. Linear Color Transfer (Reinhard et al.)
2. Correlated Color Space (Xiao and Ma)
3. Principal Component Matching (Kotera)
4. **Monge-Kantorovich Linear Color Mapping (Pitié & Kokaram) 🏆 [Best Performing]**
5. Color Distribution Transfer (Pitié et al.)
6. Singular Value Decomposition / SVD (Macenko et al.)
7. Structure-Preserving Color Normalization / SPCN (Vahadane et al.)
8. Deep Histogram Analogy (Lee et al.)
9. CAGAN (Cong et al.)

## 🗂️ Datasets
The experiments are conducted on three publicly available datasets. Please download and extract them into the `data/` directory before running the code.
* **Primary Training/Testing:** [Nickparvar WBC Dataset (based on Raabin-WBC)](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset)
* **Cross-dataset Generalization 1:** [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
* **Cross-dataset Generalization 2:** [LISC Dataset](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm)

## ⚙️ Requirements & Installation
Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## 📚 Citation

If you use this work or the evaluated color transfer algorithms, please cite:

Prakisya, N. P. T., Afiahayati, & Tyas, D. A. (2026). Color transfer for enhanced white blood cell classification in deep learning. *Informatics in Medicine Unlocked*, *66*, 101828. https://doi.org/10.1016/j.imu.2026.101828
