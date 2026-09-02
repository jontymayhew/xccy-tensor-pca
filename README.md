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

## Build the report

```bash
cd report
pandoc xccy_tensor_pca_report.md -o xccy_tensor_pca_report.docx \
  --toc --number-sections --mathml --resource-path=.
```
