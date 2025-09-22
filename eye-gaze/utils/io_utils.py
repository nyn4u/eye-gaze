import os
import pandas as pd

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path):
    return pd.read_csv(path)
