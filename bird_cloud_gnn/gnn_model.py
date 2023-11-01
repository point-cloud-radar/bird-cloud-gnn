"""Module for creating GCN class"""

import os
import dgl
import numpy as np
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GraphConv
from torch import nn
from torch import optim
from torch.nn.modules import Module
from tqdm import tqdm


os.environ["DGLBACKEND"] = "pytorch"


class GCN(nn.Module):
    """Graph Convolutional Network construction module

    A n-layer GCN is constructed from input features and list of layers
    Each layer computes new node representations by aggregating neighbour information.

    Args:
        in_feats (int): the number of input features
        layers_data (list): is a list of tuples of size of hidden layer and activation function

    Attributes:
        in_feats (int): the number of input features
        layers (nn.ModuleList): list of layers
        name (str): name of the model
        num_classes (int): the last size should correspond to the number of classes were predicting

    Methods:
        oneline_description(): Description of the model to uniquely identify it in logs
        forward(g, in_feats): Computes the output of the model.
        fit(train_dataloader, learning_rate=0.01, num_epochs=20): Train the model.
        evaluate(test_dataloader): Evaluate model.
        fit_and_evaluate(train_dataloader, test_dataloader, callback=None, learning_rate=0.01,
        num_epochs=20, sch_explr_gamma=0.99, sch_multisteplr_milestones=None,
        sch_multisteplr_gamma=0.1): Fit the model while evaluating every iteraction.
    """

    def __init__(self, in_feats: int, layers_data: list):
        """
        The __init__ function is the constructor for a class. It is called when an object of that class is instantiated.
        It can have multiple arguments and it will always be called before __new__().
        The __init__ function does not return anything.

        Args:
            self: Access variables that belongs to the class object
            in_feats: the number of input features
            layers_data: is a list of tuples of size of hidden layer and activation function

        Returns:
            The self object
        """
        super().__init__()
        self.in_feats = in_feats
        self.layers = nn.ModuleList()
        self.name = ""
        for size, activation in layers_data:
            self.layers.append(GraphConv(in_feats, size))
            self.name = self.name + f"{in_feats}-{size}_"
            in_feats = size  # For the next layer
            if activation is not None:
                assert isinstance(
                    activation, Module
                ), "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
                self.name = self.name + repr(activation).split("(", 1)[0] + "_"
            self.num_classes = size  # the last size should correspond to the number of classes were predicting

    def oneline_description(self):
        """Description of the model to uniquely identify it in logs"""
        return "-".join(["in_", f"{self.name}", "mean-out"])

    def forward(self, g, in_feats):
        """
        The forward function computes the output of the model.

        Args:
            self: Access the attributes of the class
            g: Access the graph structure and send messages between nodes
            in_feat: Pass the input feature of the node

        Returns:
            The output of the second convolutional layer
        """
        for layer in self.layers:
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                in_feats = layer(in_feats)
            else:
                in_feats = layer(g, in_feats)

        g.ndata["h"] = in_feats
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
        sch_explr_gamma=0.99,
        sch_multisteplr_milestones=None,
        sch_multisteplr_gamma=0.1,
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
            sch_explr_gamma (float): The exponential decay rate of the learning rate.
            sch_multisteplr_milestones (list): epoch numbers where the learning rate is decreased
                by a factor of sch_multisteplr_gamma. If None this is done at epoch 100
            sch_multisteplr_gamma (float): If a stepped decay of the learning rate is taken,
                the multiplication factor
        """
        if sch_multisteplr_milestones is None:
            sch_multisteplr_milestones = [min(num_epochs, 100)]
        progress_bar = tqdm(total=num_epochs)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        schedulers = [
            optim.lr_scheduler.ExponentialLR(optimizer, gamma=sch_explr_gamma),
            optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sch_multisteplr_milestones,
                gamma=sch_multisteplr_gamma,
            ),
        ]
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

            for i, pg in enumerate(optimizer.param_groups):
                epoch_values[f"LearningRate/ParGrp{i}"] = pg["lr"]
            # to visualise distribution of tensors
            for i, layer in enumerate(self.layers):
                if not isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                    epoch_values[f"Layer/conv{i}"] = layer.weight.detach()
            if self.num_classes == 2:
                epoch_values["FalseNegativeRate/test"] = num_false_negative / num_total
                epoch_values["FalsePositiveRate/test"] = num_false_positive / num_total

            progress_bar.set_postfix({"Epoch": epoch})
            progress_bar.update(1)

            for scheduler in schedulers:
                scheduler.step()

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
            shuffle=False, dataset=dataset, batch_size=batch_size, drop_last=False
        )
        labels = np.array([])
        for batched_graph, _ in dataloader:
            pred = (
                self(batched_graph, batched_graph.ndata["x"].float()).argmax(1).numpy()
            )
            labels = np.concatenate([labels, pred])
        return labels
