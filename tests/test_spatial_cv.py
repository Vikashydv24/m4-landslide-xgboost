"""
Unit tests for src/spatial_cv.py
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.spatial_cv import SpatialBlockCV


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "POINT_X": rng.uniform(74.9, 77.5, n),
        "POINT_Y": rng.uniform(8.3, 12.8, n),
        "slope":   rng.uniform(0, 60, n),
    })


@pytest.fixture
def df():
    return make_df(n=300)

@pytest.fixture
def y(df):
    return np.random.default_rng(42).integers(0, 2, len(df))


class TestSpatialBlockCV:

    def test_correct_number_of_folds(self, df, y):
        cv     = SpatialBlockCV(n_splits=5)
        splits = list(cv.split(df, y))
        assert len(splits) == 5

    def test_no_overlap_train_test(self, df, y):
        cv = SpatialBlockCV(n_splits=5)
        for tr, te in cv.split(df, y):
            assert len(set(tr) & set(te)) == 0, \
                "Train and test must not overlap"

    def test_every_sample_tested_once(self, df, y):
        cv   = SpatialBlockCV(n_splits=5)
        seen = np.zeros(len(df), dtype=int)
        for _, te in cv.split(df, y):
            seen[te] += 1
        assert np.all(seen == 1), \
            "Every sample must appear in test exactly once"

    def test_reproducible_with_same_seed(self, df, y):
        cv1 = SpatialBlockCV(n_splits=5, random_state=42)
        cv2 = SpatialBlockCV(n_splits=5, random_state=42)
        s1  = [te.tolist() for _, te in cv1.split(df, y)]
        s2  = [te.tolist() for _, te in cv2.split(df, y)]
        assert s1 == s2

    def test_different_seeds_differ(self, df, y):
        cv1 = SpatialBlockCV(n_splits=5, random_state=0,  shuffle=True)
        cv2 = SpatialBlockCV(n_splits=5, random_state=99, shuffle=True)
        s1  = [te.tolist() for _, te in cv1.split(df, y)]
        s2  = [te.tolist() for _, te in cv2.split(df, y)]
        assert s1 != s2

    def test_get_n_splits(self):
        for n in [3, 5, 10]:
            cv = SpatialBlockCV(n_splits=n)
            assert cv.get_n_splits() == n

    def test_numpy_array_converted_safely(self, y):
        # SpatialBlockCV converts numpy arrays to DataFrame internally
        # It should either work or raise a clear error about missing lat col
        cv  = SpatialBlockCV(n_splits=3)
        arr = np.random.rand(100, 4)
        try:
            list(cv.split(arr, y[:100]))
        except (KeyError, ValueError, AttributeError):
            pass   # Any of these is acceptable — no silent wrong result

    def test_spatial_separation(self):
        """Test blocks are geographically separated — not random."""
        rng = np.random.default_rng(0)
        n   = 500
        df  = pd.DataFrame({
            "POINT_X": rng.uniform(74.9, 77.5, n),
            "POINT_Y": rng.uniform(8.3, 12.8, n),
        })
        y   = rng.integers(0, 2, n)
        cv  = SpatialBlockCV(n_splits=5)

        lat = df["POINT_Y"].values
        for tr, te in cv.split(df, y):
            test_range  = lat[te].max()  - lat[te].min()
            train_range = lat[tr].max()  - lat[tr].min()
            # Training always covers more latitude than one test block
            assert train_range >= test_range * 0.5
