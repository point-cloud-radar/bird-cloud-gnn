"""Tests for the bird_cloud_gnn.my_module module.
"""
import pytest

from bird_cloud_gnn.my_module import hello


def test_hello():
    assert hello('nlesc') == 'Hello nlesc!'


def test_hello_with_error():
    with pytest.raises(ValueError) as excinfo:
        hello('nobody')
    assert 'Can not say hello to nobody' in str(excinfo.value)


@pytest.fixture
def some_name():
    return 'Jane Smith'


def test_hello_with_fixture(some_name):
    assert hello(some_name) == 'Hello Jane Smith!'
