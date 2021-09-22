import pytest
import numpy as np
import os, sys
import tensorflow as tf

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from eye.utils.utils import load_data

data_path = "/Dataset"
weight_path = "/Dataset/model_weights_resnetv2.h5"


def test_input_shape():
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data(data_path, 224)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_val.shape[0] == y_val.shape[0]


def test_input_type():
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data(data_path, 224)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(x_val, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_val, np.ndarray)


def test_input_paths():

    assert os.path.exists(
        os.path.join(data_path, "training_images.npy")
    ), "train_image Path doesn't exist"
    assert os.path.exists(
        os.path.join(data_path, "training_labels.npy")
    ), "train_label Path doesn't exist"
    assert os.path.exists(
        os.path.join(data_path, "validation_images.npy")
    ), "validation_image Path doesn't exist"
    assert os.path.exists(
        os.path.join(data_path, "validation_labels.npy")
    ), "validation_label Path doesn't exist"
    assert os.path.exists(
        os.path.join(data_path, "test_images.npy")
    ), "test_image Path doesn't exist"
    assert os.path.exists(
        os.path.join(data_path, "test_labels.npy")
    ), "test_label Path doesn't exist"


def test_weights_exist():
    assert os.path.exists(weight_path), "Weights don't exist"
