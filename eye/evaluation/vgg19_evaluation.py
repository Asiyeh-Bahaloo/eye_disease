import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from evaluation.scores import FinalScore
from evaluation.prediction import Prediction
from utils.utils import load_data, Plotter
import argparse


def vgg19_evaluation(model, x_test):
    """
    Loads the data and the model and predicts the test data.
    Parameters
    ----------
    model : models
        A model to predict the input based on that.
    x_test : numpy array
        Test images to predict the label of them.
    Returns
    -------
    y_pred : array
        Predicted labels.
    """
    print("predicting...")
    y_pred = model.predict(x_test)
    return y_pred
