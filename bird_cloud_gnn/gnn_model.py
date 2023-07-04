"""Module for creating GCN class"""

import os
import dgl
import numpy as np
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GraphConv
from torch import nn
from torch import optim
from tqdm import tqdm


os.environ["DGLBACKEND"] = "pytorch"


class GCN(nn.Module):
    """Graph Convolutional Network construction module

    A two-layer GCN is constructed from input dimension, hidden dimensions and number of classes.
    Each layer computes new node representations by aggregating neighbor information.
    """

    def __init__(self, in_feats: int, h_feats: int, num_classes: int):
        """
        The __init__ function is the constructor for a class. It is called when an object of that class is instantiated.
        It can have multiple arguments and it will always be called before __new__().
        The __init__ function does not return anything.

        Args:
            self: Access variables that belongs to the class object
            in_feats: the number of input features
            h_feats: the number of hidden features that we want to use for our first graph convolutional layer
            num_classes: the number of classes that we want to predict

        Returns:
            The self object
        """
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        """
        The forward function computes the output of the model.

        Args:
            self: Access the attributes of the class
            g: Access the graph structure and send messages between nodes
            in_feat: Pass the input feature of the node

        Returns:
            The output of the second convolutional layer
        """
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

    def fit(self, train_dataloader, learning_rate=0.01, num_epochs=20):
        """
        Train the model.

        Args:
            train_dataloader: Data loader, such as `SubsetRandomSampler`
            learning_rate (float, optional): Learning rate passed to the optimization. Defaults to 0.01.
            num_epochs (int, optional): Number of epochs of training. Defaults to 20.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for _ in range(num_epochs):
            for batched_graph, labels in train_dataloader:
                pred = self(batched_graph, batched_graph.ndata["x"].float())
                loss = nn.functional.cross_entropy(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def evaluate(self, test_dataloader):
        """
        Evaluate model.

        Args:
            test_dataloader: Data loader, such as `SubsetRandomSampler`.

        Returns:
            accuracy: Accuracy
        """
        self.eval()
        num_correct = 0
        num_tests = 0

        for batched_graph, labels in test_dataloader:
            pred = self(batched_graph, batched_graph.ndata["x"].float())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            assert pred.dim() == self.num_classes

        accuracy = num_correct / num_tests
        return accuracy

    # pylint: disable=too-many-arguments
    def fit_and_evaluate(
        self,
        train_dataloader,
        test_dataloader,
        callback=None,
        learning_rate=0.01,
        num_epochs=20,
    ):
        """Fit the model while evaluating every iteraction.

        Args:
            train_dataloader (RandomWSubsetSampler): Data loader to train set.
            test_dataloader (RandomWSubsetSampler): Data loader to test set.
            callback (callable, optional): Callback function. If defined, should receive a dict
                that stores "Loss/train", "Accuracy/train", "Loss/test", "Accuracy/test", and
                "epoch" of a single epoch. To send a stop signal, return True.
                Defaults to None.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            num_epochs (int, optional): Number of training epochs. Defaults to 20.
        """
        progress_bar = tqdm(total=num_epochs)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        epoch_values = {}
        for epoch in range(num_epochs):
            epoch_values["epoch"] = epoch
            train_loss = 0.0
            num_correct = 0
            num_total = 0
            num_false_positive = 0
            num_false_negative = 0
            self.train()
            for batched_graph, labels in train_dataloader:
                pred = self(batched_graph, batched_graph.ndata["x"].float())
                loss = nn.functional.cross_entropy(pred, labels)

                train_loss += loss.item()
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_total += len(labels)
                if self.num_classes == 2:
                    num_false_positive += (
                        ((pred.argmax(1) != labels) & (pred.argmax(1) == 1))
                        .sum()
                        .item()
                    )
                    num_false_negative += (
                        ((pred.argmax(1) != labels) & (pred.argmax(1) == 0))
                        .sum()
                        .item()
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_values["Loss/train"] = train_loss
            epoch_values["Accuracy/train"] = num_correct / num_total
            if self.num_classes == 2:
                epoch_values["FalseNegativeRate/train"] = num_false_negative / num_total
                epoch_values["FalsePositiveRate/train"] = num_false_positive / num_total

            test_loss = 0.0
            num_correct = 0
            num_total = 0
            num_false_positive = 0
            num_false_negative = 0
            self.eval()
            for batched_graph, labels in test_dataloader:
                pred = self(batched_graph, batched_graph.ndata["x"].float())

                test_loss += nn.functional.cross_entropy(pred, labels).item()
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_total += len(labels)
                if self.num_classes == 2:
                    num_false_positive += (
                        ((pred.argmax(1) != labels) & (pred.argmax(1) == 1))
                        .sum()
                        .item()
                    )
                    num_false_negative += (
                        ((pred.argmax(1) != labels) & (pred.argmax(1) == 0))
                        .sum()
                        .item()
                    )

            epoch_values["Loss/test"] = test_loss
            epoch_values["Accuracy/test"] = num_correct / num_total
            epoch_values["Layer/conv1"] = self.conv1.weight.detach()
            epoch_values["Layer/conv2"] = self.conv2.weight.detach()
            if self.num_classes == 2:
                epoch_values["FalseNegativeRate/test"] = num_false_negative / num_total
                epoch_values["FalsePositiveRate/test"] = num_false_positive / num_total

            progress_bar.set_postfix({"Epoch": epoch})
            progress_bar.update(1)

            if callback is not None:
                user_request_stop = callback(epoch_values)
                if user_request_stop is True:  # Check for explicit True
                    break

    def infer(self, dataset, batch_size=1024):
        """
        Using the model do inference on a dataset.

        Args:
            dataset: A `RadarDataSet` where for each graph inference needs to be done.

        Returns:
            labels: A numpy array with infered labels for each graph
        """
        self.eval()
        dataloader = GraphDataLoader(
            shuffle=False,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
        )
        labels = np.array([])
        for batched_graph, _ in dataloader:
            pred = (
                self(batched_graph, batched_graph.ndata["x"].float()).argmax(1).numpy()
            )
            labels = np.concatenate([labels, pred])
        return labels
