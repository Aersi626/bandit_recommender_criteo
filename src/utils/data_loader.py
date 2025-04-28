import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(
    path="data/criteo_sample.csv",
    context_features=None,
    nrows=None,
):
    """
    Loads and preprocesses the Criteo dataset for bandit simulation.

    Args:
        path (str): Path to the Criteo CSV file.
        context_features (list): List of feature names to use as context.
        nrows (int): Number of rows to load (optional).

    Returns:
        DataFrame: Preprocessed DataFrame ready for simulation.
    """
    # Default context features if none provided
    if context_features is None:
        context_features = ["I1", "I2", "C1", "C2"]

    print(f"Loading data from {path} ...")
    data = pd.read_csv(path, nrows=nrows)

    # Fill missing values
    continuous_features = [col for col in data.columns if col.startswith("I")]
    categorical_features = [col for col in data.columns if col.startswith("C")]

    data[continuous_features] = data[continuous_features].fillna(0)
    data[categorical_features] = data[categorical_features].fillna("missing")

    # Normalize continuous features
    for col in continuous_features:
        max_val = data[col].max()
        if max_val > 0:
            data[col] = data[col] / max_val

    # Label encode categorical features
    for col in categorical_features:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Create a fake 'ad_id' field (simulate 10 random ads)
    data["ad_id"] = np.random.choice(1000, size=len(data))

    # Keep only important fields
    keep_cols = ["click", "ad_id"] + context_features
    data = data[keep_cols]

    print(f"Data shape after processing: {data.shape}")
    return data