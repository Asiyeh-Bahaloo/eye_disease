from json import load
import os
import sys
import tensorflow as tf

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)


def evaluat_vgg16(model, X_test, y_test):
    """
    This function is for evaluating our VGG16 trained model using the variables passed.

    Parameters
    ----------
    model : keras.Model
        The Model we want to evaluate.
    X_test : numpy.ndarray
        Input images you want to evaluate.
    y_test : numpy.ndarray
        labels of the Input images you want to evaluate.
    """

    # display the content of the model
    baseline_results = model.evaluate(X_test, y_test, verbose=2)

    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ": ", value)
    print()
