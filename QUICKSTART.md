# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Training

```bash
python scripts/train_and_plot.py \
  --csv data/calm_water_resistance_data_CORRECTED-1.csv \
  --label Rt_N \
  --task regression
```

This will:
- ✅ Train 3 ML models (Linear Regression, Random Forest, XGBoost)
- ✅ Perform 5-fold cross-validation
- ✅ Generate 8+ publication-quality figures (PNG + SVG)
- ✅ Save best model to `models/`
- ✅ Create QR codes in `qr_codes/`

**Expected output:**
```
✓ Dataset loaded: 60 rows, 16 columns
✓ Task type detected: REGRESSION
✓ Train set: (48, 15), Test set: (12, 15)

🔄 Training models...
  ✓ Linear Regression trained
  ✓ Random Forest trained
  ✓ XGBoost trained

📊 Evaluating models...
  Linear Regression:
    CV RMSE: 264.60 ± 95.41
    Test RMSE: 264.60
    Test R²: 0.7362

  Random Forest:
    CV RMSE: 27.00 ± 8.50
    Test RMSE: 27.00
    Test R²: 0.9973

  XGBoost:
    CV RMSE: 78.70 ± 22.30
    Test RMSE: 78.70
    Test R²: 0.9767

🏆 Best model: Random Forest

💾 Model saved: models/random_forest_model.joblib

✅ All tasks complete!
```

### Step 3: View Results

```bash
# Check figures
ls figures/

# Expected output:
# correlation_heatmap.png
# correlation_heatmap.svg
# data_stats.png
# data_stats.svg
# feature_distributions.png
# feature_distributions.svg
# feature_importance.png
# feature_importance.svg
# model_cv_scores.png
# model_cv_scores.svg
# poster_ready_figure.png
# poster_ready_figure.svg
# predicted_vs_actual.png
# predicted_vs_actual.svg
# residuals_histogram.png
# residuals_histogram.svg
```

---

## 📓 Optional: Run Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Execute all cells to reproduce the same results interactively.

---

## 🔗 Upload to GitHub (Optional)

### Method 1: GitHub CLI

```bash
# 1. Edit upload_to_github.sh
# Replace REPLACE_GH_USERNAME and REPLACE_REPO_NAME

# 2. Run
bash upload_to_github.sh
```

### Method 2: Manual

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## 📱 Generate QR Codes

After uploading to GitHub:

```bash
# 1. Edit scripts/generate_qr.py
# Replace REPLACE_GH_USERNAME and REPLACE_REPO_NAME

# 2. Run
python scripts/generate_qr.py
```

---

## 🎯 What You Get

### Figures (PNG 300 DPI + SVG)
- **data_stats.png** - Dataset overview and missing values
- **feature_distributions.png** - Histograms of top 6 features
- **correlation_heatmap.png** - Feature correlation matrix
- **model_cv_scores.png** - Cross-validation comparison
- **predicted_vs_actual.png** - Scatter plot with perfect prediction line
- **residuals_histogram.png** - Error distribution
- **feature_importance.png** - Top 15 important features
- **poster_ready_figure.png** - Combined figure for presentations

### Models
- **random_forest_model.joblib** - Best trained model (reusable)

### QR Codes
- **predicted_vs_actual_qr.png** - QR to GitHub raw image
- **poster_ready_figure_qr.png** - QR to GitHub raw image
- **correlation_heatmap_qr.png** - QR to GitHub raw image

---

## ⏱️ Performance

- **Total runtime**: 30-60 seconds
- **Dataset**: 60 samples, 15 features
- **Models trained**: 3
- **Figures generated**: 8 (16 files with PNG + SVG)
- **Random seed**: 42 (fully reproducible)

---

## 🆘 Troubleshooting

**Problem**: Missing module error
```bash
pip install -r requirements.txt
```

**Problem**: CSV file not found
```bash
# Make sure CSV is in data/ directory
ls data/calm_water_resistance_data_CORRECTED-1.csv
```

**Problem**: Permission denied for upload_to_github.sh
```bash
chmod +x upload_to_github.sh
```

---

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Training script executed successfully
- [ ] 8 figures created in `figures/`
- [ ] Model saved in `models/`
- [ ] Notebook runs without errors
- [ ] (Optional) Code pushed to GitHub
- [ ] (Optional) QR codes generated

---

**You're all set! 🎉**
