"""
Main training script with model comparison and visualization.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path

from utils import (load_dataset, detect_task_type, preprocess_data, 
                   save_figure, setup_directories, RANDOM_STATE)
from preprocess import (generate_data_stats, generate_feature_distributions,
                       generate_correlation_heatmap)


def train_models(X_train, y_train):
    """Train multiple regression models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    }

    print("\n🔄 Training models...")
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  ✓ {name} trained")

    return trained_models


def evaluate_models(models, X_train, y_train, X_test, y_test, cv_folds=5):
    """Perform cross-validation and test set evaluation."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    print("\n📊 Evaluating models...")

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)

        # Test set predictions
        y_pred = model.predict(X_test)

        results[name] = {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'y_pred': y_pred
        }

        print(f"  {name}:")
        print(f"    CV RMSE: {results[name]['cv_rmse_mean']:.2f} ± {results[name]['cv_rmse_std']:.2f}")
        print(f"    Test RMSE: {results[name]['test_rmse']:.2f}")
        print(f"    Test R²: {results[name]['test_r2']:.4f}")

    return results
