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
    max_edge_distance = 5_000

    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
        max_edge_distance=max_edge_distance,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_nodes() == min_neighbours
        # max_edge_distance must be manually selected to control this
        assert min_neighbours < graph.num_edges() < min_neighbours**2
        assert label in (0, 1)

    assert dataset.has_cache()
    # Call again to run .load
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
        max_edge_distance=max_edge_distance,
    )

    # Tests that filtering to have zero graphs throws an error
    with pytest.raises(ValueError) as excinfo:
        RadarDataset(tmp_path, features, target, max_distance=0.0)
    assert "No graphs" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        RadarDataset(tmp_path, features, target, min_neighbours=10**8)
    assert "No graphs" in str(excinfo.value)

    # Test with a explicit string as argument for folder.
    dataset = RadarDataset(
        str(tmp_path),
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
    )

    # Tests that if the maximum edge distance is too small, then only self-loops are found
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
        max_edge_distance=0,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == min_neighbours

    # Tests that if the maximum edge distance is too big, then the graph is fully connected
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        max_distance=max_distance,
        min_neighbours=min_neighbours,
        max_edge_distance=max_distance**2,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == min_neighbours**2
