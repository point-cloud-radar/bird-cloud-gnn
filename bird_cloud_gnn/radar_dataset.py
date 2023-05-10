"""Module for reading the files and passing as datasets to DGL"""

import os
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
from dgl.data.utils import load_info
from dgl.data.utils import save_graphs
from dgl.data.utils import save_info
from scipy.spatial import KDTree


class RadarDataset(DGLDataset):
    """Dataset for DGL created from CSVs in a folder.

    For every labeled point in the point cloud, the number of neighbours in a specific radius is
    checked. If the number of neighbours is big enough, the point is selected. A data point is
    a graph created by taking a selected point and a number of its neighbours.


    Attributes:
        data_folder (str): Folder with the CSV files.
        features (array of str): List of features expected to be present at every CSV file.
        target (str): Target column. 0, 1 or missing expected.
        max_distance (float): Maximum distance to look for neighbours.
        min_neighbours (int): If a point has less than this amount of neighbours, it is ignored.
        max_edge_distance (float): Creates a edge between two nodes if their distance is less than this value.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        data,
        features,
        target,
        name="Radar",
        max_distance=500.0,
        min_neighbours=100,
        max_edge_distance=50.0,
    ):
        """Constructor

        Args:
            data (str or pandas.DataFrame): Folder with the CSV/parquet files, the path to a CSV/parquet file or a pandas.DataFrame.
            features (array of str): List of features expected to be present in every CSV file.
            target (str): Target column. 0, 1 or missing expected.
            max_distance (float, optional): Maximum distance to look for neighbours. Defaults to
                500.0.
            min_neighbours (int, optional): If a point has less than this amount of neighbours, it
                is ignored. Defaults to 100.
            max_edge_distance (float, optional): Creates a edge between two nodes if their distance
                is less than this value. Default to 50.0.

        Raises:
            ValueError: If `data_folder` is not a valid folder.
        """

        self.data_folder = None
        self.input_data = None
        if isinstance(data, pd.DataFrame):
            self.input_data = data
        else:
            if os.path.isdir(data) | os.path.isfile(data):
                self.data_folder = data
            else:
                raise ValueError(f"'data' is not a folder, file or pandas.DataFrame")

        self._name = name
        self.features = features
        self.target = target
        self.max_distance = max_distance
        self.min_neighbours = min_neighbours
        self.max_edge_distance = max_edge_distance
        self.graphs = []
        self.labels = []
        super().__init__(
            name=name,
            hash_key=(
                name,
                data,
                features,
                target,
                max_distance,
                min_neighbours,
                max_edge_distance,
            ),
        )

    def _read_one_file(self, data_path):
        """Reads a file and creates the graphs and labels for it."""
        split_on_dots = data_path.split(".")
        if (
            split_on_dots[-1] not in ["csv", "parquet"]
            and ".".join(split_on_dots[-2:]) != "csv.gz"
        ):
            return
        if split_on_dots[-1] == "parquet":
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path)
        self._process_data(data)

    def _process_data(self, data):
        xyz = ["x", "y", "z"]

        data = data.drop(
            data[
                np.logical_or(
                    data.range > 100000,
                    np.logical_or(data.z > 10000, data.range < 5000),
                )
            ].index
        ).reset_index(drop=True)

        data_xyz = data[xyz]
        data_features = data[self.features]

        na_index = data[data[self.target].isna()].index

        data_xyz_notna = data_xyz.drop(na_index)
        data_features_notna = data_features.drop(na_index)

        data_target = data[self.target]
        data_target_notna = data_target[data_xyz_notna.index]

        data_xyz_notna.reset_index(drop=True, inplace=True)
        data_features_notna.reset_index(drop=True, inplace=True)

        tree = KDTree(data_xyz)
        tree_notna = KDTree(data_xyz_notna)

        distance_matrix = tree_notna.sparse_distance_matrix(
            tree, self.max_distance, output_type="coo_matrix"
        )

        number_neighbours = distance_matrix.getnnz(1)
        points_of_interest = np.where(number_neighbours >= self.min_neighbours)[0]

        _, poi_indexes = tree.query(
            data_xyz_notna.loc[points_of_interest], self.min_neighbours
        )
        self.labels = np.concatenate(
            (self.labels, data_target_notna.values[points_of_interest])
        )
        for _, indexes in enumerate(poi_indexes):
            local_tree = KDTree(data_xyz.iloc[indexes])  # slow
            distances = local_tree.sparse_distance_matrix(
                local_tree, self.max_edge_distance, output_type="coo_matrix"
            )
            graph = dgl.graph((distances.row, distances.col))

            # TODO: Better fillna
            local_data = data_features.iloc[indexes].fillna(0)
            graph.ndata["x"] = torch.tensor(local_data.values)
            graph.edata["a"] = torch.tensor(distances.data)
            self.graphs.append(graph)

    def process(self):
        """Internal function for the DGLDataset. Process the folder to create the graphs."""

        self.graphs = []
        self.labels = np.array([])
        if not self.data_folder == None:
            if os.path.isdir(self.data_folder):
                for data_file in os.listdir(self.data_folder):
                    self._read_one_file(os.path.join(self.data_folder, data_file))
            else:
                if not os.path.isfile(self.data_folder):
                    raise ValueError("`data_folder` is neither a file nor a directory")
                self._read_one_file(self.data_folder)

        else:
            if not isinstance(self.input_data, pd.DataFrame):
                raise ValueError(
                    "if `self.data_folder` is not set, `self.input_data` should be a pandas.DataFrame"
                )
            self._process_data(self.input_data)

        if len(self.graphs) == 0:
            raise ValueError("No graphs selected under rules passed")
        self.labels = torch.LongTensor(self.labels)

    def save(self):
        graph_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.pkl"
        )
        save_graphs(str(graph_path), self.graphs, {"labels": self.labels})
        save_info(
            str(info_path),
            {
                "data_folder": self.data_folder,
                "features": self.features,
                "target": self.target,
                "max_distance": self.max_distance,
                "min_neighbours": self.min_neighbours,
            },
        )

    def load(self):
        graph_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.pkl"
        )
        graphs, label_dict = load_graphs(str(graph_path))
        info = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict["labels"]

        self.data_folder = info["data_folder"]
        self.features = info["features"]
        self.target = info["target"]
        self.max_distance = info["max_distance"]
        self.min_neighbours = info["min_neighbours"]

    def cache_dir(self):
        if self.data_folder == None:
            directory = self.save_dir
        else:
            directory = os.path.dirname(self.data_folder)
        return directory

    def has_cache(self):
        graph_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.cache_dir(), f"dataset_storage_{self.name}_{self.hash}.pkl"
        )
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
