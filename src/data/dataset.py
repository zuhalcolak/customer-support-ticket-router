"""Loading the processed splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(C.DATA_PROCESSED / f"{split}.csv", keep_default_na=False)
    df["y"] = df.label.map(C.LABEL2ID).astype(int)
    return df


def labels_array(df: pd.DataFrame) -> np.ndarray:
    return df["y"].to_numpy()
