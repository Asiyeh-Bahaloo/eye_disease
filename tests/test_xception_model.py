import pytest
import sys
import os
import yaml
import numpy as np

from tensorflow.python.keras.backend import dtype

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from eye.utils import utils


def test_size_load_data():
    urls = utils.read_yaml(r"tests\data_urls.yaml")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = utils.load_data(
        urls["urls_input"]
    )

    assert x_train.shape == (30, 224, 224, 3)
    assert y_train.shape == (30, 8)


def test_type_load_data():
    urls = utils.read_yaml(r"tests\data_urls.yaml")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = utils.load_data(
        urls["urls_input"]
    )

    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
