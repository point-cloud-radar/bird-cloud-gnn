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
from tqdm import tqdm


class RadarDataset(DGLDataset):
    """Dataset for DGL created from CSVs in a folder.

    For every labeled point in the point cloud, the number of neighbours in a specific radius is
    checked. If the number of neighbours is big enough, the point is selected. A data point is
    a graph created by taking a selected point and a number of its neighbours.


    Attributes:
        data_folder (str): Folder with the CSV files.
        features (array of str): List of features expected to be present at every CSV file.
        target (str): Target column. 0, 1 or missing expected.
        num_nodes (int): If a point has less than this amount of neighbours, it is ignored.
        max_edge_distance (float): Creates a edge between two nodes if their distance is less than this value.
        max_poi_per_label (int): Select at most this amount of POIs. If there are more POIs, they are chosen randomly.
    """

    missing_indicator_skip_columns = [
        "range",
        "azimuth",
        "elevation",
        "x",
        "y",
        "z",
        "centered_x",
        "centered_y",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data,
        features,
        target,
        name="Radar",
        num_nodes=100,
        max_edge_distance=50.0,
        max_poi_per_label=200,
        points_of_interest=None,
        use_missing_indicator_columns=False,
        add_edges_to_poi=False,
        skip_cache=False,
    ):
        """Constructor

        Args:
            data (str or pandas.DataFrame): Folder with the CSV/parquet files, the path to a CSV/parquet file or a pandas.DataFrame.
            features (array of str): List of features expected to be present in every CSV file.
                If "centered_x" and/or "centered_y" are included these are calculated on the fly.
            target (str): Target column. 0, 1 or missing expected.
            num_nodes (int, optional): Number of selected neighbours. Defaults to 100.
            max_edge_distance (float, optional): Creates a edge between two nodes if their distance
                is less than this value. Default to 50.0.
            points_of_interest (array of int, optional): If `data` is a pandas.Dataframe only generate graphs for these points
            use_missing_indicator_columns (bool, optional): Whether to add columns of 0s and 1s indicating values that are missing.
            add_edges_to_poi (bool, optional): Whether to add extra edges to the point of interest regardless of the edge distance.
            skip_cache (logical): If true not cache is saved to disk


        Raises:
            ValueError: If `data` is not a valid folder, file or pandas.DataFrame
            ValueError: 'points_of_interest' can only be used for pandas.Dataframe

        """

        self.data_path = None
        self.input_data = None
        if (points_of_interest is not None) and (not isinstance(data, pd.DataFrame)):
            raise ValueError(
                "'points_of_interest' can only be used for pandas.Dataframe"
            )
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
        self.num_nodes = num_nodes
        self.max_edge_distance = max_edge_distance
        self.max_poi_per_label = max_poi_per_label
        self.points_of_interest = points_of_interest
        self.use_missing_indicator_columns = use_missing_indicator_columns
        self.add_edges_to_poi = add_edges_to_poi
        if use_missing_indicator_columns:
            self.features = self.features + [
                c + "_isna"
                for c in self.features
                if c not in RadarDataset.missing_indicator_skip_columns
            ]
        self.graphs = []
        self.labels = []
        self.origin = pd.Categorical([])
        self.skip_cache = skip_cache
        super().__init__(
            name=name,
            hash_key=(
                name,
                data_hash,
                features,
                target,
                max_edge_distance,
                max_poi_per_label,
                num_nodes,
                points_of_interest,
                use_missing_indicator_columns,
                add_edges_to_poi,
            ),
        )

    def oneline_description(self):
        """Description of the dataset to uniquely identify it in logs"""
        return (
            "-".join(
                [
                    self.hash,
                    f"MED_{self.max_edge_distance}",
                    f"NN_{self.num_nodes}",
                    f"MPPL_{self.max_poi_per_label}",
                    f"UMIC_{self.use_missing_indicator_columns}",
                    f"AETP_{self.add_edges_to_poi}",
                ]
            )
            + "-features_"
            + "-".join(self.features)
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

    # pylint: disable=too-many-branches
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
        if len(data) < self.num_nodes:
            print(
                f"Warning: There are not enough points in {origin} to form neighbourhood of size {self.num_nodes}"
            )
            return

        data_xyz = data[xyz]
        # remove the special features so they can be generated later
        temp_features = self.features.copy()
        if "centered_x" in temp_features:
            temp_features.remove("centered_x")
        if "centered_y" in temp_features:
            temp_features.remove("centered_y")

        if self.use_missing_indicator_columns:
            for column in temp_features:
                if (
                    column in RadarDataset.missing_indicator_skip_columns
                    or "_isna" in column
                ):
                    continue
                data[column + "_isna"] = data[column].isna().astype("float64")

        data_features = data[temp_features].fillna(0)

        data_target = data[self.target]

        def sample_or_all(input_array, k):
            if len(input_array) <= k:
                return input_array

            rng = np.random.default_rng()
            return rng.choice(input_array, k, replace=False)

        if self.points_of_interest is not None:
            points_of_interest = self.points_of_interest
        else:
            points_of_interest = np.concatenate(
                [
                    sample_or_all(
                        data[data[self.target] == label].index.to_numpy(),
                        self.max_poi_per_label,
                    )
                    for label in [0, 1]  # Current possible labels
                ]
            )

        if self.num_nodes > 1:
            tree = KDTree(data_xyz)
            _, poi_indexes = tree.query(
                data_xyz.loc[points_of_interest], self.num_nodes
            )
        else:
            poi_indexes = np.reshape(points_of_interest, (-1, 1))
        self.labels = np.concatenate(
            (self.labels, data_target.values[points_of_interest])
        )
        for _, indexes in enumerate(poi_indexes):
            local_xyz = data_xyz.iloc[indexes].reset_index(drop=True)
            local_data = data_features.iloc[indexes].reset_index(drop=True)

            if self.num_nodes > 1:
                local_tree = KDTree(local_xyz)
                distances = local_tree.sparse_distance_matrix(
                    local_tree, self.max_edge_distance, output_type="coo_matrix"
                )
                distances_row = distances.row
                distances_col = distances.col
                distances_data = distances.data
                if self.add_edges_to_poi:
                    to_poi = local_tree.query(local_xyz.loc[0, :], self.num_nodes)
                    distances_row = np.concatenate(
                        (
                            distances_row,
                            to_poi[1],
                            np.zeros(self.num_nodes, dtype="int"),
                        )
                    )
                    distances_col = np.concatenate(
                        (
                            distances_col,
                            np.zeros(self.num_nodes, dtype="int"),
                            to_poi[1],
                        )
                    )
                    distances_data = np.concatenate(
                        (distances_data, to_poi[0], to_poi[0])
                    )
                graph = dgl.graph((distances_row, distances_col))
            else:
                distances_data = np.array([0])
                graph = dgl.graph(([0], [0]))

            # calculate special features on the fly for each graph
            if "centered_x" in self.features:
                local_data["centered_x"] = local_xyz["x"] - local_xyz.loc[0, "x"]
            if "centered_y" in self.features:
                local_data["centered_y"] = local_xyz["y"] - local_xyz.loc[0, "y"]

            local_data = local_data[self.features]

            graph.ndata["x"] = torch.tensor(local_data.values)
            graph.edata["a"] = torch.tensor(distances_data)
            graph = graph.to_simple(copy_ndata=True, copy_edata=True)
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
                print("Reading data from folder")
                data_files = sorted(os.listdir(self.data_path))
                progress_bar = tqdm(total=len(data_files))
                for data_file in data_files:
                    self._read_one_file(os.path.join(self.data_path, data_file))
                    progress_bar.set_postfix({"Data file": data_file})
                    progress_bar.update(1)
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
            print("Warning: No graphs selected under rules passed")
        self.labels = torch.LongTensor(self.labels)

    def save(self):
        if len(self.graphs) == 0:
            return
        if self.skip_cache:
            return
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
                "max_edge_distance": self.max_edge_distance,
                "max_poi_per_label": self.max_poi_per_label,
                "num_nodes": self.num_nodes,
                "origin": self.origin,
                "target": self.target,
                "points_of_interest": self.points_of_interest,
                "use_missing_indicator_columns": self.use_missing_indicator_columns,
                "add_edges_to_poi": self.add_edges_to_poi,
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
        self.max_edge_distance = info["max_edge_distance"]
        self.max_poi_per_label = info["max_poi_per_label"]
        self.num_nodes = info["num_nodes"]
        self.origin = info["origin"]
        self.target = info["target"]
        self.points_of_interest = info["points_of_interest"]
        self.use_missing_indicator_columns = info["use_missing_indicator_columns"]
        self.add_edges_to_poi = info["add_edges_to_poi"]

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
