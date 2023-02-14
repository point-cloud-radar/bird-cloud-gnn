"""Documentation about gnn-pipeline module"""

from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv

import radar_dataset
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
import importlib
importlib.reload(radar_dataset)

os.environ['DGLBACKEND'] = 'pytorch'
os.environ['TORCH'] = torch.__version__

dataset = radar_dataset.RadarDataset()
# dataset = dgl.data.GINDataset('PROTEINS', self_loop=True)

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

# Sample elements randomly without replacement.
train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False,
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False,
)

it = iter(train_dataloader)
batch = next(it)
print(batch)

batched_graph, labels = batch
print(
    "Number of nodes for each graph element in the batch:",
    batched_graph.batch_num_nodes(),
)
print(
    "Number of edges for each graph element in the batch:",
    batched_graph.batch_num_edges(),
)

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch(batched_graph)
print("The original graphs in the minibatch:")
print(graphs)

"""Say hello

    Function docstring using Google docstring style.

    Args:
        name (str): Name to say hello to

    Returns:
        str: Hello message

    Raises:
        ValueError: If `name` is equal to `nobody`

    Example:
        This function can be called with `Jane Smith` as argument using

        >>> from bird_cloud_gnn.my_module import hello
        >>> hello('Jane Smith')
        'Hello Jane Smith!'

    """

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


model = GCN(len(dataset.features), 16, 2)
# model = GCN(dataset.dim_nfeats, 16, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['x'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["x"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print("Test accuracy:", num_correct / num_tests)
