"""Module for creating GCN class"""

import os
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from torch import nn
from torch import optim


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
                assert pred.dim() == self.num_classes

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
