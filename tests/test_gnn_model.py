"""Tests for gnn_model module"""
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
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

    model = GCN(len(dataset_fixture.features), 16, 2)
    model.fit(train_dataloader)
    model.evaluate(test_dataloader)

    assert len(model.infer(dataset_fixture, batch_size=30)) == len(dataset_fixture)
    assert (
        (model.infer(dataset_fixture) == 1) | (model.infer(dataset_fixture) == 0)
    ).all()


class TestBasicBehaviour:
    """Set of tests for field access, inequality of classes and expected exceptions"""

    def test_field_access(self):
        """Test field access"""
        model = GCN(in_feats=10, h_feats=16, num_classes=2)
        assert model.in_feats == 10
        assert model.h_feats == 16
        assert model.num_classes == 2

    def test_inequality(self):
        """Test inequality of created GCN classes"""
        model1 = GCN(in_feats=10, h_feats=16, num_classes=2)
        model2 = GCN(in_feats=15, h_feats=16, num_classes=5)
        assert model1 != model2
