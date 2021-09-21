import os
import sys
import numpy as np
import yaml

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from eye.utils.utils import load_data, read_yaml


def test_output_type():

    data = read_yaml("tests/data_urls.yaml")
    data_folder = data["data_folder"]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_folder)

    assert isinstance(X_train, np.ndarray), "X_train must be a Numpy array"
    assert isinstance(y_train, np.ndarray), "y_train must be a Numpy array"
    assert isinstance(X_val, np.ndarray), "X_val must be a Numpy array"
    assert isinstance(y_val, np.ndarray), "y_val must be a Numpy array"
    assert isinstance(X_test, np.ndarray), "X_test must be a Numpy array"
    assert isinstance(y_test, np.ndarray), "y_test must be a Numpy array"


def test_output_shape():

    data = read_yaml("tests/data_urls.yaml")
    data_folder = data["data_folder"]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_folder)

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Row numbers of X and y data must be identical"
    assert (
        X_val.shape[0] == y_val.shape[0]
    ), "Row numbers of X and y data must be identical"
    assert (
        X_test.shape[0] == y_test.shape[0]
    ), "Row numbers of X and y data must be identical"

    assert (
        X_train.shape[-1] == X_val.shape[-1] == X_test.shape[-1]
    ), "All inputs must have the same #Channel"
