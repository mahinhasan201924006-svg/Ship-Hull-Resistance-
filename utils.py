"""
Utility functions for data preprocessing and visualization.
Author: ML Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fixed random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['figures', 'models', 'qr_codes']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Directories created")


def load_dataset(csv_path, label_col=None):
    """
    Load dataset from CSV file.

    Args:
        csv_path: Path to CSV file
        label_col: Name of target column (auto-detect if None)

    Returns:
        df: Loaded DataFrame
        target_col: Name of target column
    """
    df = pd.read_csv(csv_path)

    if label_col is None:
        target_col = df.columns[-1]
        print(f"⚠ Auto-detected target column: {target_col}")
    else:
        target_col = label_col

    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, target_col


def detect_task_type(y, task_type='AUTO'):
    """
    Detect if task is regression or classification.

    Args:
        y: Target variable
        task_type: 'AUTO', 'regression', or 'classification'

    Returns:
        task: 'regression' or 'classification'
    """
    if task_type.upper() != 'AUTO':
        return task_type.lower()

    # Auto-detection logic
    n_unique = y.nunique()
    if n_unique < 20 and y.dtype == 'object':
        task = 'classification'
    elif n_unique < 20 and all(y == y.astype(int)):
        task = 'classification'
    else:
        task = 'regression'

    print(f"✓ Task type detected: {task.upper()}")
    return task


def preprocess_data(df, target_col, test_size=0.2):
    """
    Preprocess data: handle missing values, encode categoricals, split.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of test set

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Handle missing values
    print(f"Missing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
    X = X.fillna(X.median(numeric_only=True))

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


def save_figure(fig, filename, formats=['png', 'svg']):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename without extension
        formats: List of formats to save
    """
    for fmt in formats:
        filepath = f"figures/{filename}.{fmt}"
        fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  ✓ Saved: figures/{filename}")
