from pathlib import Path
from tempfile import TemporaryDirectory

from bird_cloud_gnn.fake import generate_data
from bird_cloud_gnn.radar_dataset import RadarDataset
import pytest


@pytest.fixture()
def setup_dataset():
    """Setup radar dataset"""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i in range(0, 5):
            generate_data(tmp_path / f"data{i:03}.csv", 2 ** 6)

        features = [
            "range",
            "azimuth",
            "elevation",
            "x",
            "y",
            "z",
            "feat1",
            "feat2",
            "feat3",
        ]

        target = "class"
        max_distance = 30_000
        min_neighbours = 20

        dataset = RadarDataset(
            tmp_path,
            features,
            target,
            max_distance=max_distance,
            min_neighbours=min_neighbours,
        )
        return dataset
