from json import load
import os
import sys
import tensorflow as tf

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)


def evaluat_vgg16(model, X_test):
    """
    This function is for evaluating our VGG16 trained model using the variables passed.

    Parameters
    ----------
    model : keras.Model
        The Model we want to evaluate.
    X_test : numpy.ndarray
        Input images you want to evaluate.
    """

    # display the content of the model
    test_predictions_baseline = model.predict(X_test)
    return test_predictions_baseline
