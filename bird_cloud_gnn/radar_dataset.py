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
                If "centered_x" and/or "centered_y" are included these are calculated on the fly.
            target (str): Target column. 0, 1 or missing expected.
            max_distance (float, optional): Maximum distance to look for neighbours. Defaults to
                500.0.
            min_neighbours (int, optional): If a point has less than this amount of neighbours, it
                is ignored. Defaults to 100.
            max_edge_distance (float, optional): Creates a edge between two nodes if their distance
                is less than this value. Default to 50.0.

        Raises:
            ValueError: If `data` is not a valid folder, file  or pandas.DataFrame
        """

        self.data_path = None
        self.input_data = None
        if isinstance(data, pd.DataFrame):
            self.input_data = data
            data_hash = pd.util.hash_pandas_object(data).sum()
        elif os.path.isdir(data) or os.path.isfile(data):
            self.data_path = data
            data_hash = data
        else:
            raise ValueError(
                "'data' argument must be a folder, file or pandas.DataFrame"
            )

        self._name = name
        self.features = features
        self.target = target
        self.max_distance = max_distance
        self.min_neighbours = min_neighbours
        self.max_edge_distance = max_edge_distance
        self.graphs = []
        self.labels = []
        self.origin = pd.Categorical([])
        super().__init__(
            name=name,
            hash_key=(
                name,
                data_hash,
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
        self._process_data(data, origin=data_path)

    def _process_data(self, data, origin=""):
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
        # remove the special features so they can be generated later
        temp_features = self.features.copy()
        if "centered_x" in temp_features:
            temp_features.remove("centered_x")
        if "centered_y" in temp_features:
            temp_features.remove("centered_y")

        data_features = data[temp_features]

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
            local_xyz = data_xyz.iloc[indexes]
            local_tree = KDTree(local_xyz)  # slow
            distances = local_tree.sparse_distance_matrix(
                local_tree, self.max_edge_distance, output_type="coo_matrix"
            )
            graph = dgl.graph((distances.row, distances.col))

            # TODO: Better fillna
            local_data = data_features.iloc[indexes].fillna(0)
            # calculate special features on the fly for each graph
            if "centered_x" in self.features:
                local_data["centered_x"] = local_xyz[xyz[0]] - local_xyz[xyz[0]].iloc[0]
            if "centered_y" in self.features:
                local_data["centered_y"] = local_xyz[xyz[1]] - local_xyz[xyz[1]].iloc[0]

            # ensure column order is the same as in self.features
            local_data = local_data[self.features]

            graph.ndata["x"] = torch.tensor(local_data.values)
            graph.edata["a"] = torch.tensor(distances.data)
            self.graphs.append(graph)
        if origin == "":
            origin = pd.util.hash_pandas_object(data).to_string()
        self.origin = pd.api.types.union_categoricals(
            [self.origin, pd.Categorical([origin]).repeat(poi_indexes.shape[0])]
        )

    def process(self):
        """Internal function for the DGLDataset. Process the folder to create the graphs."""

        self.graphs = []
        self.labels = np.array([])
        if self.data_path is not None:
            if os.path.isdir(self.data_path):
                for data_file in os.listdir(self.data_path):
                    self._read_one_file(os.path.join(self.data_path, data_file))
            elif os.path.isfile(self.data_path):
                self._read_one_file(self.data_path)
            else:
                raise ValueError("`data_path` is neither a file nor a directory")

        elif self.input_data is not None:
            self._process_data(self.input_data)
        else:
            raise ValueError(
                "Missing input. Either self.data_path or self.input_data needs to be defined."
            )

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
                "data_path": self.data_path,
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

        self.data_path = info["data_path"]
        self.features = info["features"]
        self.target = info["target"]
        self.max_distance = info["max_distance"]
        self.min_neighbours = info["min_neighbours"]

    def cache_dir(self):
        if self.data_path is None:
            directory = self.save_dir
        elif os.path.isdir(self.data_path):
            directory = self.data_path
        elif os.path.isfile(self.data_path):
            directory = os.path.dirname(self.data_path)
        else:
            raise ValueError(
                "Missing input. Either self.data_path or self.input_data needs to be defined."
            )
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
