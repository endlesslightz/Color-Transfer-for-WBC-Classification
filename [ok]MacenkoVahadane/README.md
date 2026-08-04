# Macenko & Vahadane Color Transfer

Stain normalization (color transfer with reference) for histopathology images using:

- **Macenko** (SVD in optical-density space)
- **Vahadane** (sparse NMF / SNMF in optical-density space)

Reference stain nuance is taken from `images/may-grunwald.jpg`.

## Structure

```
MacenkoVahadane/
├── images/
│   ├── may-grunwald.jpg         # reference (May-Grünwald stain)
│   ├── Train/                   # source images (processed)
│   │   └── <class>/…jpg
│   └── Val/
│       └── <class>/…jpg
├── output/
│   ├── macenko/                 # same tree as under images/
│   └── vahadane/
├── stain_normalization/
│   ├── macenko.py
│   ├── vahadane.py
│   └── utils.py
├── main.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Keep the reference at `images/may-grunwald.jpg`. Put source histology images in subfolders under `images/` (e.g. `Train/`, `Val/`).
2. Run:

```bash
python main.py
```

Options:

```bash
python main.py --method macenko
python main.py --method vahadane
python main.py --method both
python main.py --reference images/may-grunwald.jpg --images images --output output
```

If only the reference exists, the script runs a self-normalization demo so you can verify the pipeline.

## Theory (brief)

Both methods work in **optical density (OD)** space via the Beer–Lambert law:

`OD = -log(I / I0)`

| Method | Stain estimation | Transfer idea |
|--------|------------------|---------------|
| Macenko | SVD → angular extreme vectors | Rebuild source with reference stain matrix + scaled concentrations |
| Vahadane | Sparse NMF (W stains, H concentrations) | Keep source H structure, replace W with reference W |

See also: [Understanding Stain Normalization in Histopathology](https://medium.com/@olanrewaju.charles2014/understanding-stain-normalization-in-histopathology-reinhard-macenko-and-vahadane-methods-1cce4a5a6f8e).
