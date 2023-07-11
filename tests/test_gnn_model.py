"""Tests for gnn_model module"""
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from bird_cloud_gnn.callback import CombinedCallback
from bird_cloud_gnn.callback import EarlyStopperCallback
from bird_cloud_gnn.callback import TensorboardCallback
from bird_cloud_gnn.gnn_model import GCN


def test_gnn_model(dataset_fixture):
    """Test GNN model"""
    num_examples = len(dataset_fixture)
    num_train = int(num_examples * 0.8)

    # Sample elements randomly without replacement.
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset_fixture,
        sampler=train_sampler,
        batch_size=5,
        drop_last=False,
    )
    test_dataloader = GraphDataLoader(
        dataset_fixture,
        sampler=test_sampler,
        batch_size=5,
        drop_last=False,
    )

    model = GCN(len(dataset_fixture.features), [(16, nn.ReLU()), (2, None)])
    model.fit(train_dataloader)
    model.evaluate(test_dataloader)

    callback = callback = CombinedCallback(
        [
            TensorboardCallback(),
            EarlyStopperCallback(patience=3),
        ]
    )
    model.fit_and_evaluate(train_dataloader, test_dataloader, callback)

    assert len(model.infer(dataset_fixture, batch_size=30)) == len(dataset_fixture)
    assert (
        (model.infer(dataset_fixture) == 1) | (model.infer(dataset_fixture) == 0)
    ).all()


class TestBasicBehaviour:
    """Set of tests for field access, inequality of classes and expected exceptions"""

    def test_field_access(self):
        """Test field access"""
        model = GCN(in_feats=10, layers_data=[(16, nn.ReLU()), (2, None)])
        assert model.in_feats == 10
        assert model.name == "10-16_ReLU_16-2_"
        assert model.num_classes == 2

    def test_inequality(self):
        """Test inequality of created GCN classes"""
        model1 = GCN(in_feats=10, layers_data=[(16, nn.ReLU()), (2, None)])
        model2 = GCN(in_feats=15, layers_data=[(16, nn.ReLU()), (2, None)])
        assert model1 != model2

    def test_inequality_activation(self):
        """Test inequality of created GCN classes with different activation"""
        model1 = GCN(in_feats=10, layers_data=[(16, nn.ReLU()), (2, None)])
        model2 = GCN(in_feats=10, layers_data=[(16, nn.ELU()), (2, None)])
        assert model1 != model2
        assert model2.name == "10-16_ELU_16-2_"
