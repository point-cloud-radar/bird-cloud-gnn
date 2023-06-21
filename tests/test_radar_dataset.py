"""Test RadarDataset"""

import gzip
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from bird_cloud_gnn.fake import generate_data
from bird_cloud_gnn.radar_dataset import RadarDataset


def test_radar_dataset(tmp_path):
    """Basic tests for RadarDataset"""

    with pytest.raises(ValueError) as excinfo:
        RadarDataset("nowhere", [], "")
    assert "argument must be a folder, file or pandas.DataFrame" in str(excinfo.value)

    for i in range(0, 5):
        generate_data(tmp_path / f"data{i:03}.csv", 2**6)

    with open(tmp_path / "data000.csv", "rb") as f_csv:
        with gzip.open(tmp_path / "data000.csv.gz", "wb") as f_zip:
            shutil.copyfileobj(f_csv, f_zip)
    os.remove(tmp_path / "data000.csv")

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
    num_neighbours = 20
    max_edge_distance = 5_000

    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_neighbours=num_neighbours,
        max_edge_distance=max_edge_distance,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_nodes() == num_neighbours
        assert label in (0, 1)
    assert np.array(dataset.labels).size == dataset.origin.size
    assert Path(dataset.origin[0]) == (tmp_path) / "data000.csv.gz"
    assert Path(dataset.origin[dataset.origin.size - 1]) == (tmp_path) / "data004.csv"

    assert dataset.has_cache()
    # Call again to run .load
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_neighbours=num_neighbours,
        max_edge_distance=max_edge_distance,
    )
    assert len(dataset.origin) == len(dataset)

    # Test with a explicit string as argument for folder.
    dataset = RadarDataset(
        str(tmp_path),
        features,
        target,
        num_neighbours=num_neighbours,
    )

    # Tests that if the maximum edge distance is too small, then only self-loops are found
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_neighbours=num_neighbours,
        max_edge_distance=0,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == num_neighbours

    # Tests that if the maximum edge distance is too big, then the graph is fully connected
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_neighbours=num_neighbours,
        max_edge_distance=np.inf,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == num_neighbours**2

    # Test if reading a file or a pandas.DataFrame end up with the same graph's read (both labels and graphs should correspond)
    dataset = RadarDataset(
        os.path.join(tmp_path, "data001.csv"),
        features,
        target,
        num_neighbours=num_neighbours / 2,
        max_edge_distance=max_edge_distance,
    )
    dataset_pandas = RadarDataset(
        pd.read_csv(os.path.join(tmp_path, "data001.csv")),
        features,
        target,
        num_neighbours=num_neighbours / 2,
        max_edge_distance=max_edge_distance,
    )
    assert len(dataset) == len(dataset_pandas)
    for i in range(0, len(dataset)):
        assert torch.equal(
            dataset.graphs[i].ndata["x"], dataset_pandas.graphs[i].ndata["x"]
        )
    assert torch.equal(dataset.labels, dataset_pandas.labels)


def test_manually_defined_file(tmp_path):
    # The data 'two_clusters_one_nan_one_labeled' contains a cluster around 0,
    # with all of the points unlabeled, and a cluster around (5,5,5), with
    # all of the points labeled. The points are (5,5,5), (6,5,5), (5,6,5),
    # and (5,5,6). The labels are 0 for (5,5,5) and 1 for the others.
    # There is a single feature, f1, whose value is equal to index+1.

    with open(
        tmp_path / "two_clusters_one_nan_one_labeled.csv", "w", encoding="utf-8"
    ) as f:
        f.write(
            """range,x,y,z,f1,target
10000,1,1,1,1,
10000,0,1,1,2,
10000,1,0,1,3,
10000,1,1,0,4,
10000,5,5,5,5,0
10000,6,5,5,6,1
10000,5,6,5,7,1
10000,5,5,6,8,1"""
        )

    # No constraints on max_poi
    for num_neighbours, max_poi_per_label, labels in [
        (4, 4, [0, 1, 1, 1]),
        (4, 1, [0, 1]),
        (4, 8, [0, 1, 1, 1]),
        (2, 4, [0, 1, 1, 1]),
        (2, 2, [0, 1, 1]),
        (6, 4, [0, 1, 1, 1]),
        (8, 4, [0, 1, 1, 1]),
    ]:
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_neighbours=num_neighbours,
            max_edge_distance=1.0,
            max_poi_per_label=max_poi_per_label,
        )
        assert len(dataset) == len(labels)
        assert [x[1] for x in dataset] == labels
        for graph, _ in dataset:
            assert graph.num_nodes() == num_neighbours
            if num_neighbours == 4:
                F = np.array(graph.ndata["x"])
                F = F[F[:, -1].argsort()]
                assert np.all(
                    F
                    == np.array(
                        [
                            [5, 5, 5, 5],
                            [6, 5, 5, 6],
                            [5, 6, 5, 7],
                            [5, 5, 6, 8],
                        ]
                    )
                )
            if num_neighbours <= 4:
                assert graph.num_edges() == (num_neighbours - 1) * 2 + num_neighbours
            else:
                N = num_neighbours - 4
                assert graph.num_edges() == 3 * 2 + 4 + (N - 1) * 2 + N

    # # Increase edge radius to make it a complete graph
    for num_neighbours in range(2, 8):
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_neighbours=num_neighbours,
            max_edge_distance=100.0,
            max_poi_per_label=10,
        )
        assert len(dataset) == len(labels)
        assert [x[1] for x in dataset] == labels
        for graph, _ in dataset:
            assert graph.num_nodes() == num_neighbours
            assert graph.num_edges() == num_neighbours**2


def test_centering_points(tmp_path):
    # The data 'two_clusters_one_nan_one_labeled' contains a cluster around 0,
    # with all of the points unlabeled, and a cluster around (5,5,5), with
    # all of the points labeled. The points are (5,5,5), (6,5,5), (5,6,5),
    # and (5,5,6). The labels are 0 for (5,5,5) and 1 for the others.
    # There is a single feature, f1, whose value is equal to index+1.

    with open(
        tmp_path / "two_clusters_one_nan_one_labeled.csv", "w", encoding="utf-8"
    ) as f:
        f.write(
            """range,x,y,z,f1,target
10000,1,1,1,1,
10000,1,0,0,2,
10000,0,1,0,3,
10000,0,0,1,4,
10000,5,5,5,5,0
10000,6,5,5,6,1
10000,5,6,5,7,1
10000,5,5,6,8,1"""
        )

    dataset = RadarDataset(
        tmp_path,
        ["x", "centered_y", "f1"],
        "target",
        num_neighbours=8,
    )

    graph, label = dataset[0]
    assert label == 0
    F = np.array(graph.ndata["x"])
    assert np.all(
        F[F[:, 2].argsort()]
        == np.array(
            [
                [1, -4, 1],
                [1, -5, 2],
                [0, -4, 3],
                [0, -5, 4],
                [5, 0, 5],
                [6, 0, 6],
                [5, 1, 7],
                [5, 0, 8],
            ]
        )
    )
    with pytest.raises(KeyError) as excinfo:
        RadarDataset(
            tmp_path,
            ["x", "y", "z", "f2"],
            "target",
            num_neighbours=8,
        )
    assert "['f2'] not in index" in str(excinfo.value)


def test_no_graphs(tmp_path):
    with open(tmp_path / "no_graphs.csv", "w", encoding="utf-8") as f:
        f.write(
            """range,x,y,z,f1,target
10000,1,1,1,1,
10000,0,1,1,2,
10000,1,0,1,3,
10000,1,1,0,4,"""
        )

    dataset = RadarDataset(
        tmp_path,
        ["x", "centered_y", "f1"],
        "target",
        num_neighbours=8,
    )
    assert len(dataset) == 0
