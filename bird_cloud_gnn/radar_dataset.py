import os

os.environ['DGLBACKEND'] = 'pytorch'
import torch

import dgl
from dgl.data import DGLDataset

import pandas as pd
import numpy as np
from scipy.spatial import KDTree


class RadarDataset(DGLDataset):
    """Preprocess data and create a dgl graph instance.

    This is class description.

    Parameters
    ----------
    Attributes
    ----------

    """

    def __init__(self):
        super().__init__(name="Radar")

    def process(self):
        print(os.getcwd())
        self.graphs = []
        self.labels = []

        data_folder = 'data/manual_annotations'
        self.features = ['range', 'azimuth', 'elevation', 'x', 'y', 'z', 'DBZH', 'DBZV']
        self.target = 'BIOLOGY'
        max_distance = 500
        # min_neighbours = 200
        # TODO: handle exception when min_neighbours for some datasets is small
        min_neighbours = 100
        # TODO: Expand to all files
        data_file = os.listdir(data_folder)[0]
        df = pd.read_csv(os.path.join(data_folder, data_file))
        df = df.drop(
            df[np.logical_or(df.range > 100000, np.logical_or(df.z > 10000, df.range < 5000))].index).reset_index(
            drop=True)
        df_notna = df.drop(df[df[self.target].isna()].index).reset_index(drop=True)
        tree = KDTree(df.loc[:, ['x', 'y', 'z']])
        tree_notna = KDTree(df_notna.loc[:, ['x', 'y', 'z']])
        distance_matrix = tree_notna.sparse_distance_matrix(tree, max_distance)
        number_neighbours = np.array(np.sum(distance_matrix > 0, axis=1)).reshape(-1)
        points_of_interest = np.where(number_neighbours >= min_neighbours)[0]

        for p in points_of_interest:
            _, indexes = tree.query(df.loc[p, ['x', 'y', 'z']], min_neighbours)
            local_tree = KDTree(df.loc[indexes, ['x', 'y', 'z']])
            D = local_tree.sparse_distance_matrix(local_tree, max_distance, output_type='coo_matrix')
            # Create a graph
            g = dgl.graph((D.row, D.col))
            # TODO: Better fillna
            local_df = df.loc[indexes, self.features].fillna(0)
            assert not np.any(np.isnan(local_df))
            g.ndata["x"] = torch.tensor(local_df.values)
            g.edata["a"] = torch.tensor(D.data)
            self.graphs.append(g)
            self.labels.append(df_notna.loc[p, self.target])
        # Convert the label list to tensor
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

