# Cross-Currency Curve Analysis with Tensor PCA

A quantum-computing use-case test proposal: build USD-collateralized OIS curves
(USD, EUR, GBP, JPY), generate a zero-rate scenario tensor, and compare classic
(flattened) PCA against structure-preserving Tensor PCA (Tucker / HOSVD).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Execute the notebooks in order:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_*.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_*.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_Tensor_PCA.ipynb --inplace
```

Notebook 03 exports figures to `report/figures/`.




## Reproducibility

- **Quick start (no ORE):** the precomputed `data/zero_rate_tensor.npz` is
  included — just run `notebooks/03_Tensor_PCA.ipynb` to reproduce the PCA /
  Tensor-PCA analysis and figures.
- **Full rebuild (needs ORE):** to regenerate the tensor from market data, run
  notebooks `01` and `02` first (`open-source-risk-engine` from
  `requirements.txt` — no source build required).