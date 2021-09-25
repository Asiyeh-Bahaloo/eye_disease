import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import argparse
from utils.utils import MlflowCallback


def vgg19_train(
    model, x_train, y_train, x_val, y_val, batch_size=8, epochs=3, patience=5
):
    """
    Loads the data, compiles the model, and trains the model.
    Parameters
    ----------
    model : model
        raw model to be trained.
    x_train : numpy array
        Training images to fit the model on them.
    y_train : numpy array
        Training labels to fit the model on them.
    x_val : numpy array
        Validation images to validate the model based on them.
    y_val : numpy array
        Validation labels to validate the model based on them.
    batch_size : int
        Batch size fr training the model.
    epochs : int
        Number of epochs to train.
    patience : int
        Number of patience for early stopping.
    Returns
    -------
    model : models
        Model that has fitted to the data.
    history : dict
        The history of the training.
    """
    EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=1
    )

    print("fitting model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[EarlyStoppingCallback, MlflowCallback()],
    )

    return model, history
