"""Tests for cross_validation"""
from bird_cloud_gnn.cross_validation import (
    kfold_evaluate,
    leave_one_origin_out_evaluate,
)
from torch import nn


def test_kfold_evaluate(dataset_fixture):
    """Test cross validation with KFold"""

    kfold_evaluate(
        dataset_fixture,
        layers_data=[(32, nn.ReLU()), (2, None)],
    )


def test_leave_one_out_evaluate(dataset_fixture):
    """Test cross validation with leave-one-out"""

    leave_one_origin_out_evaluate(
        dataset_fixture,
        layers_data=[(32, nn.ReLU()), (2, None)],
    )
