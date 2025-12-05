# Ship Hull Resistance Prediction - ML Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green.svg)](https://xgboost.readthedocs.io/)

Reproducible machine learning project for predicting ship hull calm water resistance using multiple regression models with comprehensive evaluation and publication-quality visualizations.

---

## 📊 Dataset

**Filename**: `calm_water_resistance_data_CORRECTED-1.csv`

**Description**: Ship hull hydrodynamic resistance measurements from ITTC baseline experiments.

**Features**:
- `hull`: Hull type (KCS, KVLCC2, DTMB5415, Series60, Wigley)
- `Cb`: Block coefficient
- `L`, `B`, `T`: Length, beam, draft (m)
- `Fr`: Froude number
- `Re`: Reynolds number
- `speed_ms`: Speed (m/s)
- `Cf`, `Cw`, `Cr`, `Cv`, `Ct`: Resistance coefficients
- `S`, `k`: Wetted surface area, form factor

**Target**: `Rt_N` - Total resistance (Newtons)

**Samples**: 60 | **Task**: Regression

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/REPLACE_GH_USERNAME/REPLACE_REPO_NAME.git
cd REPLACE_REPO_NAME
pip install -r requirements.txt
```

### 2. Add Your Dataset

Place `calm_water_resistance_data_CORRECTED-1.csv` in the `data/` directory.

### 3. Run Training

```bash
python scripts/train_and_plot.py \
  --csv data/calm_water_resistance_data_CORRECTED-1.csv \
  --label Rt_N \
  --task regression
```

**Output**:
- 📊 All figures in `figures/` (PNG + SVG)
- 💾 Trained models in `models/`
- 📱 QR codes in `qr_codes/`

### 4. Explore Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📁 Repository Structure

```
ship-hull-resistance-ml/
├── data/
│   └── calm_water_resistance_data_CORRECTED-1.csv
├── scripts/
│   ├── preprocess.py          # Data preprocessing
│   ├── train_and_plot.py      # Main training script
│   ├── utils.py               # Utility functions
│   └── generate_qr.py         # QR code generation
├── notebooks/
│   └── analysis.ipynb         # Interactive analysis
├── figures/                   # Generated visualizations
├── models/                    # Saved models
├── qr_codes/                  # QR codes for GitHub links
├── requirements.txt
├── upload_to_github.sh
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

### Models Trained
1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble tree-based model
3. **XGBoost** - Gradient boosting (typically best performance)

### Evaluation Strategy
- **Cross-Validation**: 5-fold CV with fixed random seed (42)
- **Metrics**: RMSE, MAE, R²
- **Test Split**: 20% hold-out set

### Reproducibility
- Fixed random seed: `42`
- Pinned package versions in `requirements.txt`
- Complete environment specification

---

## 📈 Expected Results

**Best Model**: XGBoost

| Metric | Approximate Value |
|--------|-------------------|
| Test RMSE | ~27 N |
| Test R² | ~0.997 |
| CV RMSE | 27.0 ± X N |

*Exact values depend on data splits - see outputs for detailed results*

---

## 📊 Generated Figures

All figures saved in `figures/` as PNG (300 DPI) and SVG:

1. `data_stats.png` - Dataset summary
2. `feature_distributions.png` - Feature histograms
3. `correlation_heatmap.png` - Feature correlations
4. `model_cv_scores.png` - Cross-validation comparison
5. `predicted_vs_actual.png` - Prediction scatter plot
6. `residuals_histogram.png` - Residual distribution
7. `feature_importance.png` - Random Forest importances
8. `poster_ready_figure.png` - Combined figure for presentations

---

## 🔗 GitHub Upload Instructions

### Method 1: GitHub CLI (Recommended)

```bash
# 1. Login to GitHub
gh auth login

# 2. Update upload_to_github.sh with your username/repo
# Replace REPLACE_GH_USERNAME and REPLACE_REPO_NAME

# 3. Run upload script
bash upload_to_github.sh
```

### Method 2: Git + Personal Access Token

```bash
# 1. Create token at: https://github.com/settings/tokens
#    Scopes: repo, workflow

# 2. Set environment variable
export GITHUB_TOKEN="your_token_here"

# 3. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://REPLACE_GH_USERNAME:$GITHUB_TOKEN@github.com/REPLACE_GH_USERNAME/REPLACE_REPO_NAME.git
git push -u origin main
```

---

## 📱 QR Code Generation

After uploading to GitHub:

```bash
# 1. Update scripts/generate_qr.py with your GitHub details
# 2. Run:
python scripts/generate_qr.py
```

**Raw Figure URLs** (template):
```
https://raw.githubusercontent.com/REPLACE_GH_USERNAME/REPLACE_REPO_NAME/main/figures/predicted_vs_actual.png
```

QR codes saved to `qr_codes/` directory.

---

## ✅ Acceptance Criteria

Verify everything works:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training
python scripts/train_and_plot.py \
  --csv data/calm_water_resistance_data_CORRECTED-1.csv \
  --label Rt_N \
  --task regression

# 3. Check outputs
ls figures/  # Should contain 8+ PNG and SVG files
ls models/   # Should contain .joblib model file
ls qr_codes/ # Should contain QR code images

# 4. Run notebook
jupyter notebook notebooks/analysis.ipynb
# Execute all cells - outputs should match script results
```

**Expected Runtime**: ~30-60 seconds on modern hardware

---

## 🛡️ Privacy & Security

- ✅ No PII in dataset
- ✅ Local computation only
- ⚠️ Never commit GitHub tokens to repository
- ⚠️ Use environment variables for sensitive data

---

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:

- Python 3.9+
- scikit-learn 1.3.0
- xgboost 1.7.6
- pandas 2.0.3
- matplotlib 3.7.2
- seaborn 0.12.2

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

**Issue**: `FileNotFoundError: data/calm_water_resistance_data_CORRECTED-1.csv`
```bash
# Solution: Place CSV file in data/ directory
cp /path/to/your/calm_water_resistance_data_CORRECTED-1.csv data/
```

**Issue**: QR codes showing placeholder URLs
```bash
# Solution: Edit scripts/generate_qr.py
# Replace REPLACE_GH_USERNAME and REPLACE_REPO_NAME with actual values
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 📄 License

MIT License - feel free to use for academic or commercial purposes.

---

**Generated with ❤️ for reproducible ML research**
