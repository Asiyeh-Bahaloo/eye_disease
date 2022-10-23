import pytest
from tensorflow.keras import Sequential
from tensorflow.python.keras.engine.functional import Functional
from eye.models.Inception_V3 import Inception_V3


num_classes = 8

def test_model():
    model = Inception_V3.inception(8)
    assert  isinstance(model, Sequential) or isinstance(model, Functional)

