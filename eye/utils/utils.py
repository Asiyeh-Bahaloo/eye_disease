import os
import sys
import mlflow
import tensorflow as tf
import numpy as np
import yaml
import csv
import matplotlib.pyplot as plt


curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from eye.evaluation.metrics import *


def read_yaml(dir):
    with open(dir) as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def load_data(data_path="/Data"):
    """
    Loads the ODIR dataset.

    Parameters
    ----------
    data_path : str, optional
        Direct Path to Input data folder, by default "/Data"
        You must save these 6 file in this folder:
        - training_images.npy
        - training_labels.npy
        - validation_images.npy
        - validation_labels.npy
        - test_images.npy
        - test_labels.npy

    Returns
    -------
    Train_tuple : tuple
        this tuple includes (X_train, y_train)
    Validation_tuple : tuple
        this tuple includes (X_val, y_val)
    Test_tuple : tuple
        this tuple includes (X_test, y_test)
    """
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

    X_train = np.load(os.path.join(data_path, "training_images.npy"))
    y_train = np.load(os.path.join(data_path, "training_labels.npy"))

    X_val = np.load(os.path.join(data_path, "validation_images.npy"))
    y_val = np.load(os.path.join(data_path, "validation_labels.npy"))

    X_test = np.load(os.path.join(data_path, "test_images.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_weights(model, save_folder):
    """
    This function saves the weights of the model in a (.h5) file.

    Parameters
    ----------
    model : keras.Model
        The model you want to save its weights.
    save_folder : str
        Direct path to the file you want to save your weights on.
    """
    model.save(save_folder)


def print_metrics(gt, pred, threshold=0.5):
    """
    This function prints all the metrics we have.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5
    """
    kappa = kappa_score(gt, pred, threshold)
    f1 = f1_score(gt, pred, threshold)
    auc = auc_score(gt, pred)
    final = final_score(gt, pred, threshold)
    print("Kappa score is: ", kappa)
    print("f1 score is: ", f1)
    print("auc score is: ", auc)
    print("final score is: ", final)


def save_predict_output(predictions, path):
    """
    This function help you to save your model's prediction vector.
    The final file would be prediction.csv located in the path given to this function.

    Examples
    --------
    >>> y_pred = model.predict(x_test)
    >>> save_predict_output(y_pred, path_to_result_folder/name.csv)

    Parameters
    ----------
    predictions : numpy.ndarray
        The model's output vector.
    path : str
        path to the file you want to save for predictions.
    """

    try:
        with open(path, "w", newline="") as csv_file:
            file_writer = csv.writer(
                csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            file_writer.writerow(
                [
                    "ID",
                    "Normal",
                    "Diabetes",
                    "Glaucoma",
                    "Cataract",
                    "AMD",
                    "Hypertension",
                    "Myopia",
                    "Others",
                ]
            )
            count = 0
            for sub in predictions:
                normal = sub[0]
                diabetes = sub[1]
                glaucoma = sub[2]
                cataract = sub[3]
                amd = sub[4]
                hypertension = sub[5]
                myopia = sub[6]
                others = sub[7]
                file_writer.writerow(
                    [
                        count,
                        normal,
                        diabetes,
                        glaucoma,
                        cataract,
                        amd,
                        hypertension,
                        myopia,
                        others,
                    ]
                )
                count = count + 1
    except:
        msg = "couldn't open the provided path"
        print(msg)


class MlflowCallback(tf.keras.callbacks.Callback):

    # This function will be called after each epoch.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        # Log the metrics from Keras to MLflow
        mlflow.log_metric("loss", logs["loss"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)
        mlflow.log_metric("precision", logs["precision"], step=epoch)
        mlflow.log_metric("val_precision", logs["val_precision"], step=epoch)
        mlflow.log_metric("recall", logs["recall"], step=epoch)
        mlflow.log_metric("val_recall", logs["val_recall"], step=epoch)
        mlflow.log_metric("auc", logs["auc"], step=epoch)
        mlflow.log_metric("val_auc", logs["val_auc"], step=epoch)

    # This function will be called after training completes.
    def on_train_end(self, logs=None):
        mlflow.log_param("num_layers", len(self.model.layers))
        mlflow.log_param("optimizer_name", type(self.model.optimizer).__name__)

    plt.show()
