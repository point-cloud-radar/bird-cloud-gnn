"""Tests for cross_validation"""
from bird_cloud_gnn.cross_validation import kfold_evaluate


def test_kfold_evaluate(dataset_fixture):
    """Test cross validation with KFold"""

    kfold_evaluate(
        dataset_fixture,
        h_feats=32,
    )
