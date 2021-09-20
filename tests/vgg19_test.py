import pytest
import numpy as np
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from eye.utils.utils import load_data


def test_data_type():
    data_folder = os.path.join("..", "data")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(data_folder)

    assert isinstance(x_train, np.ndarray), "x_train must be a numpy array"
    assert isinstance(x_train, np.ndarray), "y_train must be a numpy array"

    assert isinstance(x_val, np.ndarray), "x_val must be a numpy array"
    assert isinstance(x_val, np.ndarray), "y_val must be a numpy array"

    assert isinstance(x_test, np.ndarray), "x_test must be a numpy array"
    assert isinstance(x_test, np.ndarray), "y_test must be a numpy array"


def test_data_shape():
    data_folder = os.path.join("..", "data")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(data_folder)

    assert (
        x_train.shape[1:] == x_val.shape[1:] == x_test.shape[1:]
    ), "Input shapes are incompatible"
    assert (
        y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:]
    ), "Input shapes are incompatible"
