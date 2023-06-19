import logging
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from bird_cloud_gnn.gnn_model import GCN


def kFoldEvaluate(
    dataset, n_splits=5, learning_rate=0.01, num_epochs=20, batch_size=512
):
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

        learning_rate = 0.01
        model = GCN(len(dataset.features), 16, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(num_epochs):
            model.train()
            for batched_graph, labels in train_dataloader:
                pred = model(batched_graph, batched_graph.ndata["x"].float())
                loss = torch.nn.functional.cross_entropy(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                assert pred.dim() == model.num_classes

        model.eval()
        num_correct = 0
        num_tests = 0

        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata["x"].float())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            accuracy = num_correct / num_tests
            assert pred.dim() == model.num_classes
            logging.info("Fold %s - Test accuracy: %s", fold + 1, accuracy)

        progress_bar.set_postfix({"Fold": fold + 1})
        progress_bar.update(1)
