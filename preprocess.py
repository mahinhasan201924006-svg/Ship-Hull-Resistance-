"""
Data preprocessing and exploratory analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset, save_figure, setup_directories


def generate_data_stats(df):
    """Generate dataset summary statistics figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dataset info
    stats_text = f"""
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━
Rows: {df.shape[0]:,}
Columns: {df.shape[1]}
Numeric: {df.select_dtypes(include=[np.number]).shape[1]}
Categorical: {df.select_dtypes(include=['object']).shape[1]}
Total Missing: {df.isnull().sum().sum():,}
    """

    axes[0].text(0.1, 0.5, stats_text, fontsize=14, 
                 family='monospace', va='center')
    axes[0].axis('off')
    axes[0].set_title('Dataset Overview', fontsize=16, fontweight='bold')

    # Missing values per column
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    if missing.sum() > 0:
        missing.plot(kind='barh', ax=axes[1], color='coral')
        axes[1].set_title('Top 10 Columns with Missing Values', fontsize=14)
        axes[1].set_xlabel('Missing Count')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values ✓', 
                    ha='center', va='center', fontsize=16)
        axes[1].axis('off')

    plt.tight_layout()
    save_figure(fig, 'data_stats')
    plt.close()


def generate_feature_distributions(df, n_features=6):
    """Generate distribution plots for top numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:n_features]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', 
                      edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_figure(fig, 'feature_distributions')
    plt.close()


def generate_correlation_heatmap(df):
    """Generate correlation matrix heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'correlation_heatmap')
    plt.close()


if __name__ == "__main__":
    setup_directories()
    df, target_col = load_dataset('data/calm_water_resistance_data_CORRECTED-1.csv')

    generate_data_stats(df)
    generate_feature_distributions(df)
    generate_correlation_heatmap(df)

    print("\n✓ Preprocessing complete!")
