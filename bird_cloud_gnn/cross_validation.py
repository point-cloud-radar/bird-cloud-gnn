"""Helper functions for cross validation.
"""
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from bird_cloud_gnn.gnn_model import GCN


# pylint: disable=too-many-arguments, too-many-locals
def kfold_evaluate(
    dataset,
    h_feats=16,
    n_splits=5,
    learning_rate=0.01,
    num_epochs=100,
    batch_size=512,
):
    """
    Evaluate the model on a dataset using StratifiedKFold.

    Args:
        dataset (RadarDataset): The dataset
        h_feats (int, optional): The number of hidden features of the model
        n_splits (int, optional): Number of folds. Defaults to 5.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
        num_epochs (int, optional): Training epochs. Defaults to 20.
        batch_size (int, optional): Batch size used in the data loaders. Defaults to 512.
    """

    labels = np.array(dataset.labels)
    # Initialize a stratified k-fold splitter
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    progress_bar = tqdm(total=n_splits)
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, labels)):
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

        model = GCN(
            in_feats=len(dataset.features),
            h_feats=h_feats,
            num_classes=2,
        )
        model.fit(train_dataloader, learning_rate=learning_rate, num_epochs=num_epochs)

        accuracy = model.evaluate(test_dataloader)
        print(f"Fold {fold + 1} - Test accuracy: {accuracy}")

        progress_bar.set_postfix({"Fold": fold + 1})
        progress_bar.update(1)
