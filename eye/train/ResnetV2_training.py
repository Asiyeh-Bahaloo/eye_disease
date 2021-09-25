import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import MlflowCallback

defined_metric = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

num_classes = 8


def resnet_v2_training(
    x_train, y_train, x_val, y_val, model, batch_size, epochs, patience
):

    """[A function for training resnet_v2 model]

    Parameters
    ----------
    x_train : [np.ndarray]
        [training images]
    y_train : [np.ndarray]
        [training labels]

    x_val : [np.ndarray]
        [validation images]
    y_val : [np.ndarray]
        [validation labels]

    model : [type]
        [Model object from the script ResnetV2.py]
    defined_metrics : [type], optional
        [description], by default defined_metric
    batch_size : int, optional
         by default 32
    epochs : int, optional
        [number of epochs], by default 5
    patience : int, optional
        [number of patience for early stopping], by default 5

    results_path :  [str]
        [The address which will saves our weights]
    """

    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        validation_split=0.2,
        callbacks=[earlyStoppingCallback, MlflowCallback()],
    )

    return history
