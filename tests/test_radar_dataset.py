"""Test RadarDataset"""

import pytest
from bird_cloud_gnn.fake import generate_data
from bird_cloud_gnn.radar_dataset import RadarDataset


def test_radar_dataset(tmp_path):
    """Basic tests for RadarDataset"""

    with pytest.raises(ValueError) as excinfo:
        RadarDataset("nowhere", [], "")
    assert "not a folder" in str(excinfo.value)

    for i in range(0, 5):
        generate_data(tmp_path / f"data{i:03}.csv", 2**6)

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
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_nodes() == min_neighbours
        assert label in (0, 1)

    assert dataset.has_cache()
    # Call again to run .load
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
    )

    with pytest.raises(ValueError) as excinfo:
        RadarDataset(tmp_path, features, target, max_distance=0.0)
    assert "No graphs" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        RadarDataset(tmp_path, features, target, min_neighbours=10**8)
    assert "No graphs" in str(excinfo.value)
