import os
import mlflow
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import yaml
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from eye.evaluation.metrics import kappa_score, f1_score, auc_score, final_score


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


def calc_metrics(gt, pred, threshold=0.5):
    """
    This function calculates all the metrics we have.

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

    return {"kappa": kappa, "f1": f1, "auc": auc, "final": final}


def pprint_metrics(scores):
    """pretty prints scores dictionary

    Parameters
    ----------
    scores : dict
        A dictionary containing scores with it's keys being score names
    """
    for key, value in scores.items():
        print(f"    {key} score is : ", value)


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
        mlflow.log_metric("specificity", logs["specificity"], step=epoch)
        mlflow.log_metric("val_specificity", logs["val_specificity"], step=epoch)

        metrics = [
            "loss",
            "accuracy",
            "precision",
            "recall",
            "kappa",
            "f1",
            "auc",
            "final",
            "specificity",
        ]
        labels = ["N", "D", "G", "C", "A", "H", "M", "O"]

        for metric in metrics:
            for label in labels:
                mlflow.log_metric(
                    label + "_" + metric, logs[metric + "ForLabel" + label], step=epoch
                )
                mlflow.log_metric(
                    "val_" + label + "_" + metric,
                    logs["val_" + metric + "ForLabel" + label],
                    step=epoch,
                )

    # This function will be called after training completes.
    def on_train_end(self, logs=None):
        mlflow.log_param("num_layers", len(self.model.layers))
        mlflow.log_param("optimizer_name", type(self.model.optimizer).__name__)


def find_similar_images(indices, filenames, result_folder):
    """[summary]

    Parameters
    ----------
    indices : [type]
        [description]
    filenames : list
        list of all training images names
    result_folder : str
        The address which you want to place similar images
    """
    plt.figure(figsize=(15, 10), facecolor="white")
    plotnumber = 1
    for index in indices:
        if plotnumber < len(indices):
            ax = plt.subplot(2, 4, plotnumber)
            plt.imshow(mpimg.imread(filenames[index]), interpolation="lanczos")
            plt.imsave(
                result_folder + "/" + str(plotnumber) + ".jpg",
                mpimg.imread(filenames[index]),
            )
            plotnumber += 1
    plt.tight_layout()


def split_ODIR(path_train_val, train_val_frac=0.8, random_state=2021):
    """
    This function splits and returns train, val dataframes of ODIR dataset.

    Parameters
    ----------
    path_train_val : str
        path to the .csv file contains the label mapping for train, val sections.
    train_val_frac : float, optional
        fraction you want to use in train, val split, by default 0.8
    random_state : int, optional
        random_state used during this function, by default 2021

    Returns
    -------
    Tuple of Two DataFrames
        (df_train, df_val)
    """

    df = (
        pd.read_csv(path_train_val)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    split_point = math.ceil(train_val_frac * len(df))
    df_train = df[:split_point].reset_index(drop=True)
    df_val = df[split_point:].reset_index(drop=True)

    return (df_train, df_val)


def split_Cataract(Image_path, train_val_frac=0.8, random_state=2021):
    """
    This function splits and returns train, val dataframes of Cataract dataset.

    Parameters
    ----------
    Image_path : str
        Path to the image folder of the cataract dataset. this folder must have 4 folders in it:
        - 1_normal
        - 2_cataract
        - 2_glaucoma
        - 3_retina_disease
    train_val_frac : float, optional
        fraction you want to use in train, val split, by default 0.8
    random_state : int, optional
        random_state used during this function, by default 2021

    Returns
    -------
    Tuple of Two DataFrames
        (df_train, df_val)
    """

    df = pd.DataFrame(columns=["ID", "normal", "cataract", "glaucoma", "others"])

    for idx1, filename in enumerate(os.listdir(os.path.join(Image_path, "1_normal"))):
        df.loc[idx1] = [filename, 1, 0, 0, 0]
    for idx2, filename in enumerate(os.listdir(os.path.join(Image_path, "2_cataract"))):
        df.loc[idx2 + idx1 + 1] = [filename, 0, 1, 0, 0]
    for idx3, filename in enumerate(os.listdir(os.path.join(Image_path, "2_glaucoma"))):
        df.loc[idx3 + idx2 + idx1 + 2] = [filename, 0, 0, 1, 0]
    for idx4, filename in enumerate(
        os.listdir(os.path.join(Image_path, "3_retina_disease"))
    ):
        df.loc[idx4 + idx3 + idx2 + idx1 + 3] = [filename, 0, 0, 0, 1]

    # shuffle dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_point = math.ceil(train_val_frac * len(df))

    df_train = df[:split_point].reset_index(drop=True)
    df_val = df[split_point:].reset_index(drop=True)

    return (df_train, df_val)


def add_args_to_mlflow(args):
    """
    This function logs all arguments of argparse in the mlflow.

    Parameters
    ----------
    args : Namespace
        a namespace containing the arguments of argparse
        
    Returns
    -------
    logs each argument in the mlflow
    """
    for arg in args.__dict__:
        mlflow.log_param(arg, args.__dict__[arg])
