from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from bird_cloud_gnn.fake import generate_data
from bird_cloud_gnn.radar_dataset import RadarDataset


@pytest.fixture()
def dataset_fixture(feat_fixture):
    """Setup radar dataset"""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i in range(0, 5):
            generate_data(tmp_path / f"data{i:03}.csv", 2**6)

        num_nodes = 20
        features = feat_fixture["features"]
        target = feat_fixture["target"]
        dataset = RadarDataset(
            tmp_path,
            features,
            target,
            num_nodes=num_nodes,
            max_poi_per_label=100,
        )
        return dataset


@pytest.fixture()
def feat_fixture():
    """Features and target"""
    feat = {
        "features": [
            "range",
            "azimuth",
            "elevation",
            "x",
            "y",
            "z",
            "feat1",
            "feat2",
            "feat3",
        ],
        "target": "class",
    }

    return feat
