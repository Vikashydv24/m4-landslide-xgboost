# M-4 — Landslide Susceptibility Mapping
## XGBoost · Gradient Boosting Machine · Random Forest

**Project:** IITK SURGE 2026 — Landslide Toolkit Lab
**Student:** Vikash Kumar Yadav · NIT Warangal
**Mentor:** Dr. Shyam Nandan · Department of Earth Science · IIT Kanpur
**Dataset:** Hao 2020 Kerala 2018 monsoon landslide inventory (DS-1, 4,728 events)
**Reference:** Sahin (2020) *SN Applied Sciences* 2:1308

---

## Overview

This repository reproduces model M-4 of the IITK Landslide Toolkit project —
a comparative assessment of three ensemble tree-based machine learning algorithms
(XGBoost, Gradient Boosting Machine, Random Forest) for landslide susceptibility
mapping in Kerala, India, using the 2018 monsoon event dataset.

**Key design choices:**
- **Spatial block cross-validation** (5 folds, latitude bands) — prevents data
  leakage from spatial autocorrelation and gives honest AUC estimates
- **11 terrain predictors** derived from SRTM 30m DEM (no land-use leak)
- **SHAP analysis** for feature importance and model interpretability
- **Wilcoxon signed-rank test** for statistical significance between models
- **Land-masked LSM maps** — Arabian Sea excluded from predictions

---

## Results

### Model Performance (5-fold Spatial Block CV)

| Model | OA (Mean±Std) | AUC (Mean±Std) | RMSE (Mean±Std) | Kappa (Mean±Std) |
|-------|--------------|----------------|-----------------|------------------|
| **RF** | **0.8313 ± 0.0407** | **0.8861 ± 0.0249** | 0.3527 ± 0.0355 | 0.5064 ± 0.2297 |
| GBM | 0.8166 ± 0.0322 | 0.8818 ± 0.0243 | 0.3680 ± 0.0353 | 0.4762 ± 0.2106 |
| XGBoost | 0.8210 ± 0.0376 | 0.8829 ± 0.0279 | 0.3633 ± 0.0363 | 0.4884 ± 0.2250 |

### Key Findings
- RF achieves the highest AUC (0.8861), consistent with Sahin (2020) RF_opt (AUC=0.8860)
- Wilcoxon signed-rank test: **no statistically significant difference** between models (p>0.05)
- All three models perform equivalently on Kerala 2018 terrain data
- SHAP analysis: **elevation and slope** are the dominant susceptibility drivers
- ~40% of Kerala land area classified as High or Very High susceptibility
- High-risk zones concentrated along the **Western Ghats escarpment** (Idukki, Wayanad, Kozhikode)

---

## Project Structure
---

## Predictors Used (11 terrain features from SRTM 30m DEM)

| Category | Features |
|---|---|
| Topographic | Elevation, Slope, Aspect, Plan Curvature, Profile Curvature, TRI, TPI* |
| Hydrological | TWI, SPI, STI, Flow Accumulation, Distance to River |

*TPI excluded (all-NaN after extraction — known DEM edge effect)

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/Vikashydv24/m4-landslide-xgboost.git
cd m4-landslide-xgboost

conda env create -f environment_clean.yml
conda activate m4-lsm
```

### 2. Get the data

Kerala 2018 landslide inventory:
> van Westen, C.J. (2020). DOI: 10.17026/dans-x6c-y7x2

Place shapefile in `data/raw/`. Download SRTM DEM for bbox
(74.8–77.6°E, 8.2–13.0°N) and place as `data/dem/kerala_srtm30.tif`.

### 3. Run the pipeline

```bash
# Download DEM
python scripts/00_download_dem.py

# Derive 12 terrain predictors
python scripts/01_derive_predictors.py

# Build training dataset
python scripts/02_prepare_samples.py

# Train RF / GBM / XGBoost + SHAP + figures
python scripts/03_train_models.py

# Produce LSM maps
python scripts/05_produce_lsm_map.py

# Run unit tests
python -m pytest tests/ -v
```

---

## Susceptibility Map — Kerala 2018 (XGBoost)

Results show the Western Ghats escarpment as the dominant
high-susceptibility zone, consistent with the 2018 disaster pattern.

| Class | Area Coverage |
|---|---|
| Very Low | 20.0% |
| Low | 20.0% |
| Moderate | 20.0% |
| High | 20.0% |
| Very High | 20.0% |

---

## Unit Tests — 22/22 Passing

---

## References

1. Sahin, E.K. (2020). Assessing the predictive capability of ensemble
   tree methods for landslide susceptibility mapping using XGBoost,
   gradient boosting machine, and random forest.
   *SN Applied Sciences* **2**, 1308.
   https://doi.org/10.1007/s42452-020-3060-1

2. Pham, B.T., Pradhan, B., Bui, D.T., Prakash, I., & Dholakia, M.B.
   (2016). A comparative study of different machine learning methods
   for landslide susceptibility assessment: a case study of Uttarakhand
   area (India). *Environmental Modelling & Software* **84**, 240–250.
   https://doi.org/10.1016/j.envsoft.2016.07.005

---

## License

Code: MIT.
Data: CC-BY 4.0 (van Westen, University of Twente, 2020).
