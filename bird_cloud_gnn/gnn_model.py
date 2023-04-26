"""Module for creating GCN class"""
import logging
import os
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from torch import nn


os.environ["DGLBACKEND"] = "pytorch"

# FIX retrieve module name instead of __main__
# This might not be relevant if all logs are written to the same file?
# for now hardcoded
MODULE_NAME = "gnn_model"
# module_name = os.path.splitext(os.path.basename(__name__))[0]
log_filename = os.path.join("../logs/", f"{MODULE_NAME}.log")


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

        # set up logging
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_filename, mode="w")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.info("Instantiating %s", type(self).__name__)
        # maybe change this later if too much output
        self.logger.info("%s", self.__repr__())

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


# # testing
# if __name__ == "__main__":
#     model = GCN(in_feats=10, h_feats=16, num_classes=2)
