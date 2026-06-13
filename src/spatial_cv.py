"""
Spatial Block Cross-Validation for LSM.
Splits Kerala into N latitude bands.
Each fold withholds one band as test set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
import warnings


class SpatialBlockCV(BaseCrossValidator):

    def __init__(self, n_splits=5, lat_col="POINT_Y",
                 lon_col="POINT_X", shuffle=True, random_state=42):
        self.n_splits     = n_splits
        self.lat_col      = lat_col
        self.lon_col      = lon_col
        self.shuffle      = shuffle
        self.random_state = random_state

    def _assign_blocks(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with lat/lon columns.")
        lat   = X[self.lat_col].values
        edges = np.linspace(lat.min(), lat.max(), self.n_splits + 1)
        ids   = np.digitize(lat, edges[:-1]) - 1
        return np.clip(ids, 0, self.n_splits - 1)

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        block_ids = self._assign_blocks(X)
        unique    = np.unique(block_ids)

        if self.shuffle:
            rng    = np.random.default_rng(self.random_state)
            unique = rng.permutation(unique)

        idx = np.arange(len(X))
        for b in unique:
            test_idx  = idx[block_ids == b]
            train_idx = idx[block_ids != b]
            if len(test_idx) > 0 and len(train_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
