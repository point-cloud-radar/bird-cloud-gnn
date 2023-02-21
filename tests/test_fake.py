"""Tests for bird_cloud_gnn.fake module."""

import numpy as np
import pandas as pd
from bird_cloud_gnn.fake import generate_data


def test_generate_data(tmp_path):
    """Tests basic information about the generated data"""
    filename = tmp_path / "fake_data.csv"
    df = generate_data(
        filename=filename,
        num_points=2**4,
        min_range=0.1,
        max_range=2.3,
        azimuth_skip=90.0,
        elevations=[0, 30, 60],
    )
    assert df.shape == (2**4, 11)
    assert df.range.min() >= 0.1
    assert df.range.max() <= 2.3
    assert sorted(df.azimuth.unique()) == [0.5, 90.5, 180.5, 270.5]
    assert sorted(df.elevation.unique()) == [0, 30, 60]
    for feat in ["feat1", "feat2", "feat3"]:
        assert df[feat].min() >= 0
        assert df[feat].max() <= 1
    assert (
        df.columns
        == [
            "range",
            "azimuth",
            "elevation",
            "useless_feature",
            "x",
            "y",
            "z",
            "feat1",
            "feat2",
            "feat3",
            "class",
        ]
    ).all()
    assert df.notna().all(axis=None)
    df2 = pd.read_csv(filename)

    assert (np.abs(df2 - df) < 1e-12).all(
        axis=None
    )  # Round off error might occur saving the file!

    df = generate_data(
        filename=None,
        add_na=True,
    )
    assert df.isna().any(axis=None)
