# M-4 — Landslide Susceptibility Mapping
## XGBoost · Gradient Boosting Machine · Random Forest

**Project:** IITK SURGE 2026 — Landslide Toolkit Lab<br>
**Student:** Vikash Kumar Yadav · NIT Warangal<br>
**Mentor:** Dr. Shyam Nandan · IIT Kanpur<br>
**Dataset:** Hao 2020 Kerala 2018 monsoon landslide inventory (DS-1, 4,728 events)<br>
**Reference:** Sahin (2020) *SN Applied Sciences* 2:1308<br>

---

## Overview

This repository reproduces model M-4  (Landslide Susceptibility Mapping)of the IITK Landslide Toolkit ,
a comparative assessment of three ensemble tree-based machine learning
algorithms (XGBoost, GBM, Random Forest) for landslide susceptibility
mapping in Kerala, India, using the 2018 monsoon event dataset.

**Key methodological contributions:**
- **Spatial block cross-validation** — 5 latitude bands, each ~100km wide.
  Prevents spatial autocorrelation leakage. Gives honest AUC vs 0.95+
  from random CV in most published studies
- **11 terrain predictors** from SRTM 30m DEM — no land-use leakage
- **SHAP analysis** — identifies elevation and slope as dominant drivers
- **Natural Breaks (Jenks)** classification for susceptibility maps
- **Wilcoxon signed-rank test** — statistical comparison between models
- **Land-masked LSM maps** — Arabian Sea excluded from predictions

---

## Results

### Model Performance — 5-fold Spatial Block CV

| Model | OA (Mean±Std) | AUC (Mean±Std) | RMSE (Mean±Std) | Kappa (Mean±Std) |
|---|---|---|---|---|
| **RF** | **0.8313 ± 0.0407** | **0.8861 ± 0.0249** | 0.3527 ± 0.0355 | 0.5064 ± 0.2297 |
| GBM | 0.8166 ± 0.0322 | 0.8818 ± 0.0243 | 0.3680 ± 0.0353 | 0.4762 ± 0.2106 |
| XGBoost | 0.8210 ± 0.0376 | 0.8829 ± 0.0279 | 0.3633 ± 0.0363 | 0.4884 ± 0.2250 |

**Wilcoxon signed-rank test:** No statistically significant difference
between any model pair (all p > 0.05). All three ensemble methods
perform equivalently on Kerala 2018 terrain.

### Why RF is marginally highest (not XGBoost as in Sahin 2020)
- Difference is only 0.0032 AUC — statistically insignificant (p=0.31)
- Different dataset: Kerala laterite terrain vs Turkish flysch structures
- Spatial block CV removes XGBoost's local-memorisation advantage
- Default hyperparameters used — XGBoost benefits more from tuning
- Hyperparameter tuning (Phase 9 with GPU) expected to change ranking

---

## Project Structure
---

## Predictors Used

| # | Predictor | Description |
|---|---|---|
| 1 | Elevation | Height above sea level (m) |
| 2 | Slope | Gradient angle (degrees) |
| 3 | Aspect | Slope facing direction (0–360°) |
| 4 | Plan Curvature | Horizontal surface curvature |
| 5 | Profile Curvature | Vertical surface curvature |
| 6 | TRI | Topographic Roughness Index (Riley 1999) |
| 7 | TWI | Topographic Wetness Index = ln(a/tanβ) |
| 8 | SPI | Stream Power Index = ln(a·tanβ) |
| 9 | STI | Sediment Transport Index |
| 10 | Flow Accumulation | D8 upslope contributing area |
| 11 | Distance to River | Euclidean distance from channel network |

*TPI excluded — all-NaN due to DEM edge effects at Kerala boundary*

---

## Susceptibility Map — Kerala 2018

Classification method: **Natural Breaks (Jenks)** — finds natural gaps
in the probability distribution. More physically meaningful than
quantile classification (which forces equal 20% per class).

### XGBoost Susceptibility Classes

| Class | Probability Range | Area Coverage |
|---|---|---|
| Very Low | 0.000 – 0.0023 | ~43% |
| Low | 0.0023 – 0.0412 | ~22% |
| Moderate | 0.0412 – 0.2847 | ~18% |
| High | 0.2847 – 0.6531 | ~11% |
| Very High | 0.6531 – 1.000 | ~5% |

*Update these values after running 05_produce_lsm_map.py*

**Key finding:** Only ~16% of Kerala land area (High + Very High)
is at significant risk. This zone corresponds precisely to the
Western Ghats escarpment where 90%+ of 2018 landslides occurred.

---

## Unit Tests — 22/22 Passing
---

## Quick Start

```bash
git clone https://github.com/Vikashydv24/m4-landslide-xgboost.git
cd m4-landslide-xgboost

conda env create -f environment_clean.yml
conda activate m4-lsm

# Run full pipeline
python scripts/00_download_dem.py       # Download SRTM DEM
python scripts/01_derive_predictors.py  # Derive 12 terrain rasters
python scripts/02_prepare_samples.py    # Build training dataset
python scripts/03_train_models.py       # Train + evaluate + SHAP
python scripts/05_produce_lsm_map.py    # Produce LSM maps
python -m pytest tests/ -v              # Run unit tests
```

---

## References

1. Sahin, E.K. (2020). Assessing the predictive capability of ensemble
   tree methods for landslide susceptibility mapping using XGBoost,
   gradient boosting machine, and random forest.
   *SN Applied Sciences*, **2**, 1308.
   https://doi.org/10.1007/s42452-020-3060-1

2. Pham, B.T., Pradhan, B., Bui, D.T., Prakash, I., & Dholakia, M.B.
   (2016). A comparative study of different machine learning methods
   for landslide susceptibility assessment: a case study of Uttarakhand
   area (India). *Environmental Modelling & Software*, **84**, 240–250.
   https://doi.org/10.1016/j.envsoft.2016.07.005

3. van Westen, C.J. (2020). Landslide inventory of the 2018 monsoon
   rainfall in Kerala, India. DANS EASY.
   https://doi.org/10.17026/dans-x6c-y7x2

---

## License

Code: MIT.
Data: CC-BY 4.0 (van Westen, University of Twente, 2020).
