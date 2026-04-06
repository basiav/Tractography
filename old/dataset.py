from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    path: str
    maps: np.ndarray
    df: pd.DataFrame

    def __str__(self):
        return self.name.upper()