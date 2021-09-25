from json import load
import os
import sys
import tensorflow as tf


curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)
from utils.utils import MlflowCallback


def train_from_file(
    model, X_train, y_train, X_val, y_val, batch_size=2, epochs=2, patience=5
):
    """
    This function tries to train your VGG16 model based on transfer learning.

    Parameters
    ----------
    model : keras.Model
        The Model we want to train.
    X_train : numpy.ndarray
        Input images you want to train your model on.
    y_train : numpy.ndarray
        Labels of the Input images you want to train your model on.
    X_val : numpy.ndarray
        Input images you want to evaluate your model on.
    y_val : numpy.ndarray
        Labels of the Input images you want to evaluate your model on.
    batch_size : int, optional
        defines batch size of our model, by default 2
    epochs : int, optional
        defines number of epochs we want to train our model, by default 2
    patience : int, optinal
        defines number of patience in early stopping

    Returns
    -------
    keras.Model
        Trained model.
    dictionary
        History of the training process.
    """

    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,  # class_weight = class_weight,
        validation_data=(X_val, y_val),
        callbacks=[earlyStoppingCallback, MlflowCallback()],
    )
    print("model trained successfuly.")

    return model, history
