# M-4 — Landslide Susceptibility Mapping: XGBoost, GBM, Random Forest

**Project:** IITK Surge 2026 — Landslide Toolkit Lab  
**Dataset:** Hao 2020 Kerala 2018 landslide inventory (DS-1, 4,728 events)  
**Reference:** Sahin (2020) *SN Applied Sciences* 2:1308  
**Target:** Honest AUC 0.80–0.85 under spatial-block CV

---

## Overview

This repository reproduces model M-4 of the Landslide Toolkit project:
ensemble tree methods (XGBoost, Gradient Boosting, Random Forest) for
landslide susceptibility mapping in Kerala, India, using the 2018 monsoon
event dataset.

**Key design choices:**
- **Spatial-block CV** (not random CV) → prevents data leakage, honest AUC
- **SHAP** for model interpretation (feature importance)
- **Wilcoxon signed-rank test** for statistical comparison (Table 6, Sahin 2020)
- **XGBoost with L1/L2 regularisation** matches or beats RF under class imbalance

---

## Results

| Model   | OA (%)        | AUC (%)       | RMSE   | Kappa  |
|---------|---------------|---------------|--------|--------|
| RF      | — ± —         | — ± —         | —      | —      |
| GBM     | — ± —         | — ± —         | —      | —      |
| XGBoost | — ± —         | — ± —         | —      | —      |

*(filled after training)*

---

## Project Structure
---

## Quick Start

### 1. Clone & setup environment

```bash
git clone https://github.com/YOUR_USERNAME/m4-landslide-xgboost.git
cd m4-landslide-xgboost

conda env create -f environment_clean.yml
conda activate m4-lsm
```

### 2. Get the data

Download the Kerala 2018 landslide inventory from:
> van Westen, C.J. (2020). DOI: [10.17026/dans-x6c-y7x2](https://doi.org/10.17026/dans-x6c-y7x2)

Place the shapefile in `data/raw/`.

Download SRTM 30m DEM for Kerala (bbox: 74.8–77.6°E, 8.2–13.0°N):
> [SRTM tile selector](https://dwtkns.com/srtm30m/)

Place the merged DEM as `data/dem/kerala_srtm30.tif`.

### 3. Run the pipeline

```bash
# Step 1: Derive 12 terrain predictors from DEM
python scripts/01_derive_predictors.py

# Step 2: Prepare training dataset (landslide + background samples)
python scripts/02_prepare_samples.py

# Step 3: Train models + SHAP + generate all figures
python scripts/03_train_models.py
```

Results appear in `results/`.

---

## Predictors Used (15 factors, Sahin 2020)

| Category      | Factors |
|---------------|---------|
| Topographic   | Elevation, Slope, Aspect, Plan Curvature, Profile Curvature, TRI, TPI |
| Hydrological  | TWI, Drainage density, Distance to river, SPI, STI |
| Land cover    | LULC (LU_2018), NDVI |
| Geology       | Lithology *(pending)* |

---

## Key Reference

Sahin, E.K. (2020). Assessing the predictive capability of ensemble tree methods
for landslide susceptibility mapping using XGBoost, gradient boosting machine,
and random forest. *SN Applied Sciences* **2**, 1308.
https://doi.org/10.1007/s42452-020-3060-1

---

## License

Code: MIT. Data: CC-BY 4.0 (original dataset by van Westen, University of Twente).
