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
    num_nodes = 20
    max_edge_distance = 5_000

    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_nodes=num_nodes,
        max_edge_distance=max_edge_distance,
    )
    with pytest.raises(ValueError) as excinfo:
        RadarDataset(
            tmp_path,
            features,
            target,
            num_nodes=num_nodes,
            max_edge_distance=max_edge_distance,
            points_of_interest=[1, 2, 3],
        )
    assert "'points_of_interest' can only be used for pandas.Dataframe" in str(
        excinfo.value
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_nodes() == num_nodes
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
        num_nodes=num_nodes,
        max_edge_distance=max_edge_distance,
    )
    assert len(dataset.origin) == len(dataset)

    # Test with a explicit string as argument for folder.
    dataset = RadarDataset(
        str(tmp_path),
        features,
        target,
        num_nodes=num_nodes,
    )

    # Tests that if the maximum edge distance is too small, then only self-loops are found
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_nodes=num_nodes,
        max_edge_distance=0,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == num_nodes

    # Tests that if the maximum edge distance is too big, then the graph is fully connected
    dataset = RadarDataset(
        tmp_path,
        features,
        target,
        num_nodes=num_nodes,
        max_edge_distance=np.inf,
    )
    assert len(dataset) > 0
    for graph, label in dataset:
        assert graph.num_edges() == num_nodes**2

    # Test if reading a file or a pandas.DataFrame end up with the same graph's read (both labels and graphs should correspond)
    dataset = RadarDataset(
        os.path.join(tmp_path, "data001.csv"),
        features,
        target,
        num_nodes=num_nodes / 2,
        max_edge_distance=max_edge_distance,
    )
    dataset_pandas = RadarDataset(
        pd.read_csv(os.path.join(tmp_path, "data001.csv")),
        features,
        target,
        num_nodes=num_nodes / 2,
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
10000,5,6,5,,1
10000,5,5,6,8,1"""
        )

    # No constraints on max_poi
    for num_nodes, max_poi_per_label, labels in [
        (4, 4, [0, 1, 1, 1]),
        (4, 1, [0, 1]),
        (4, 8, [0, 1, 1, 1]),
        (2, 4, [0, 1, 1, 1]),
        (2, 2, [0, 1, 1]),
        (6, 4, [0, 1, 1, 1]),
        (8, 4, [0, 1, 1, 1]),
    ]:
        for use_missing_indicator_columns in [False, True]:
            dataset = RadarDataset(
                tmp_path,
                ["x", "y", "z", "f1"],
                "target",
                num_nodes=num_nodes,
                max_edge_distance=1.0,
                max_poi_per_label=max_poi_per_label,
                use_missing_indicator_columns=use_missing_indicator_columns,
            )
            assert len(dataset) == len(labels)
            assert [x[1] for x in dataset] == labels
            if use_missing_indicator_columns:
                assert dataset.features == ["x", "y", "z", "f1", "f1_isna"]
            else:
                assert dataset.features == ["x", "y", "z", "f1"]
            for graph, _ in dataset:
                assert graph.num_nodes() == num_nodes
                if num_nodes == 4:
                    expected = np.array(
                        [
                            [5, 6, 5, 0],
                            [5, 5, 5, 5],
                            [6, 5, 5, 6],
                            [5, 5, 6, 8],
                        ]
                    )
                    if use_missing_indicator_columns:
                        expected = np.concatenate(
                            (expected, np.array([[1], [0], [0], [0]])), axis=1
                        )
                    F = np.array(graph.ndata["x"])
                    F = F[F[:, 3].argsort()]
                    assert np.all(F == expected)
                if num_nodes <= 4:
                    assert graph.num_edges() == (num_nodes - 1) * 2 + num_nodes
                else:
                    N = num_nodes - 4
                    assert graph.num_edges() == 3 * 2 + 4 + (N - 1) * 2 + N

    # # Increase edge radius to make it a complete graph
    for num_nodes in range(2, 8):
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_nodes=num_nodes,
            max_edge_distance=100.0,
            max_poi_per_label=10,
        )
        assert len(dataset) == len(labels)
        assert [x[1] for x in dataset] == labels
        for graph, _ in dataset:
            assert graph.num_nodes() == num_nodes
            assert graph.num_edges() == num_nodes**2


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
        num_nodes=8,
        use_missing_indicator_columns=False,
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
            num_nodes=8,
        )
    assert "f2" in str(excinfo.value)

    # Points of interest creates dataset of expected length and changing
    #  order of point matches order of dataset
    dataset = RadarDataset(
        pd.read_csv(str(tmp_path) + "/two_clusters_one_nan_one_labeled.csv"),
        ["x", "centered_y", "f1"],
        "target",
        num_nodes=4,
        points_of_interest=[1, 2],
    )
    assert len(dataset) == 2
    dataset2 = RadarDataset(
        pd.read_csv(str(tmp_path) + "/two_clusters_one_nan_one_labeled.csv"),
        ["x", "centered_y", "f1"],
        "target",
        num_nodes=4,
        points_of_interest=[3, 1, 6],
    )
    assert len(dataset2) == 3
    graph, _ = dataset[0]
    graph2, _ = dataset2[1]

    assert np.all(np.array(graph.ndata["x"]) == np.array(graph2.ndata["x"]))
    assert np.all(np.array(graph.edata["a"]) == np.array(graph2.edata["a"]))


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
        num_nodes=8,
    )
    assert len(dataset) == 0


def test_not_enough_points_in_neighbourhood(tmp_path):
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

    dataset = RadarDataset(
        tmp_path,
        ["x", "y", "z", "f1"],
        "target",
        num_nodes=8,
        max_edge_distance=2.0,
        max_poi_per_label=10,
    )
    assert len(dataset) == 4
    dataset = RadarDataset(
        tmp_path,
        ["x", "y", "z", "f1"],
        "target",
        num_nodes=9,
        max_edge_distance=2.0,
        max_poi_per_label=10,
    )
    assert len(dataset) == 0
    dataset = RadarDataset(
        pd.read_csv(os.path.join(tmp_path, "two_clusters_one_nan_one_labeled.csv")),
        ["x", "y", "z", "f1"],
        "target",
        num_nodes=9,
        max_edge_distance=2.0,
        max_poi_per_label=10,
    )
    assert len(dataset) == 0


def test_num_nodes_equal_to_1(tmp_path):
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

    dataset = RadarDataset(
        tmp_path,
        ["x", "y", "z", "f1"],
        "target",
        num_nodes=1,
        max_edge_distance=2.0,
        max_poi_per_label=10,
    )
    assert len(dataset) == 4
    for graph, _ in dataset:
        assert graph.num_nodes() == 1


def test_add_edges_to_poi(tmp_path):
    with open(
        tmp_path / "two_clusters_one_nan_one_labeled.csv", "w", encoding="utf-8"
    ) as f:
        f.write(
            """range,x,y,z,f1,target
10000,2,2,2,1,
10000,0,2,2,2,
10000,2,0,2,3,
10000,2,2,0,4,
10000,5,5,5,5,0
10000,6,5,5,6,1
10000,5,6,5,7,1
10000,5,5,6,8,1"""
        )

    for num_nodes in range(1, 8):
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_nodes=num_nodes,
            max_edge_distance=0.0,
            max_poi_per_label=10,
            add_edges_to_poi=True,
        )
        for graph, _ in dataset:
            assert graph.num_edges() == num_nodes + 2 * (num_nodes - 1)

    for num_nodes in range(4, 8):
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_nodes=num_nodes,
            max_edge_distance=1.5,  # enough for cluster (5,5) to connect but not for cluster (0,0)
            max_poi_per_label=10,
            add_edges_to_poi=True,
        )
        for graph, _ in dataset:
            # cluster (5, 5) to each other + self + extra to poi
            assert graph.num_edges() == 4 * 3 + num_nodes + 2 * (num_nodes - 4)

    for num_nodes in range(1, 8):
        dataset = RadarDataset(
            tmp_path,
            ["x", "y", "z", "f1"],
            "target",
            num_nodes=num_nodes,
            max_edge_distance=25.0,
            max_poi_per_label=10,
            add_edges_to_poi=True,
        )
        for graph, _ in dataset:
            assert graph.num_edges() == num_nodes * num_nodes
