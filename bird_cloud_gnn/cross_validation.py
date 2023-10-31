"""Helper functions for cross validation.
"""
import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from bird_cloud_gnn.gnn_model import GCN


def get_dataloaders(dataset, train_idx, test_idx, batch_size):
    """
    Returns train and test dataloaders for a given dataset, train indices, test indices, and batch size.

    Args:
        dataset (torch_geometric.datasets): The dataset to use for creating dataloaders.
        train_idx (list): The indices to use for training.
        test_idx (list): The indices to use for testing.
        batch_size (int): The batch size to use for the dataloaders.

    Returns:
        tuple: A tuple containing the train and test dataloaders.
    """
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_dataloader = GraphDataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=False,
    )
    test_dataloader = GraphDataLoader(
        dataset=dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


# pylint: disable=too-many-arguments, too-many-locals
def kfold_evaluate(
    dataset,
    layers_data,
    n_splits=5,
    learning_rate=0.01,
    num_epochs=100,
    batch_size=512,
):
    """
    Evaluate the model on a dataset using StratifiedKFold.

    Args:
        dataset (RadarDataset): The dataset
        layers_data (list): The list of input size and activation
        n_splits (int, optional): Number of folds. Defaults to 5.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
        num_epochs (int, optional): Training epochs. Defaults to 20.
        batch_size (int, optional): Batch size used in the data loaders. Defaults to 512.

    Returns:
        None
    """
    labels = np.array(dataset.labels)
    # Initialize a stratified k-fold splitter
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    progress_bar = tqdm(total=n_splits)
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, labels)):
        train_dataloader, test_dataloader = get_dataloaders(
            dataset, train_idx, test_idx, batch_size
        )

        model = GCN(
            in_feats=len(dataset.features),
            layers_data=layers_data,
        )
        model.fit(train_dataloader, learning_rate=learning_rate, num_epochs=num_epochs)

        accuracy = model.evaluate(test_dataloader)
        print(f"Fold {fold + 1} - Test accuracy: {accuracy}")

        progress_bar.set_postfix({"Fold": fold + 1})
        progress_bar.update(1)


def leave_one_origin_out_evaluate(
    dataset,
    layers_data,
    learning_rate=0.01,
    num_epochs=100,
    batch_size=512,
):
    """
    Evaluate the model on a dataset by looping over each origin, and training the data with
    all data not from that origin, and testing with data from that origin. In other words,
    doing a leave one out validation on the origins.

    Args:
        dataset (RadarDataset): The dataset.
        layers_data (list): The list of input size and activation
        n_splits (int, optional): Number of folds. Defaults to 5.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
        num_epochs (int, optional): Training epochs. Defaults to 20.
        batch_size (int, optional): Batch size used in the data loaders. Defaults to 512.
    """
    origins = pd.Series(dataset.origin)
    unique_origins = origins.unique()
    progress_bar = tqdm(total=len(unique_origins))

    print(origins)

    for origin in unique_origins:
        train_idx = origins[origins != origin].index.to_list()
        test_idx = origins[origins == origin].index.to_list()

        train_dataloader, test_dataloader = get_dataloaders(
            dataset, train_idx, test_idx, batch_size
        )

        model = GCN(
            in_feats=len(dataset.features),
            layers_data=layers_data,
        )
        model.fit(train_dataloader, learning_rate=learning_rate, num_epochs=num_epochs)

        accuracy = model.evaluate(test_dataloader)
        print(
            f"Origin {origin} - Test accuracy: {accuracy} - Length test: {len(test_idx)}"
        )

        progress_bar.set_postfix({"Origin": origin})
        progress_bar.update(1)
