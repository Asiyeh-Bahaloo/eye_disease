import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import glob
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

from eye.models.vgg16 import Vgg16
from eye.utils.utils import pprint_metrics, calc_metrics
from eye.utils import plotter_utils as p
from eye.data.transforms import (
    Compose,
    Resize,
    RemovePadding,
    BenGraham,
    RandomShift,
    RandomFlipLR,
    RandomFlipUD,
    KerasPreprocess,
)


def parse_arguments():
    # define arguments
    parser = argparse.ArgumentParser(
        description="Arguments for evaluating the VGG16 model"
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        default="/Data/model_weights_vgg16.h5",
        help="Path to the model's weights file",
        required=True,
    )
    parser.add_argument(
        "--data",
        dest="data_folder",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )

    parser.add_argument(
        "--label",
        dest="label_file",
        type=str,
        default="/Data",
        help="Path to the model's input labels file (.csv)",
        required=True,
    )

    parser.add_argument(
        "--result",
        dest="result_path",
        type=str,
        default="/Data",
        help="Path to the folder you want to save your results",
        required=True,
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        default="binary_crossentropy",
        help="type of loss function with which you want to compile your model",
        required=False,
    )

    args = parser.parse_args()
    return args


# python scripts/eval_vgg16.py --weights=./Data/model_weights_vgg16.h5 --data=./Data --result=./Data
def main():
    args = parse_arguments()

    # Parameters
    num_classes = 8
    tag = "vgg16"

    # Load data
    compose_test = Compose(
        transforms=[
            RemovePadding(),
            # BenGraham(350),
            Resize((224, 224), False),
            KerasPreprocess(model_name="vgg16"),
            # RandomShift(0.2, 0.3),
            # RandomFlipLR(),
            # RandomFlipUD(),
        ]
    )

    # reading test labels dataframe
    df = pd.read_csv(args.label_file)
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    os.chdir(args.data_folder)

    # temp list for storing images
    X_test_ls = []
    Y_test_ls = []

    for idx, filename in enumerate(tqdm(sorted(glob.glob("*.jpg")))):

        # handling image
        img = cv2.imread(filename)
        img = compose_test(img)
        X_test_ls.append(img)

        # handling label
        if idx % 2 == 0:
            image_id = int(filename[: filename.find("_")])
            Y_test_ls.append(df.loc[df["ID"] == image_id, class_names].to_numpy())

    img_shape = (len(X_test_ls),) + (X_test_ls[0].shape)
    label_shape = (len(Y_test_ls),) + (Y_test_ls[0].shape[1],)

    X_test = np.stack(X_test_ls, axis=0).reshape(img_shape)
    Y_test = np.stack(Y_test_ls, axis=0).reshape(label_shape)

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Model
    model = Vgg16(num_classes=num_classes, input_shape=(224, 224, 3))
    model.load_weights(path=args.weights_path)

    # predicting image labels
    test_predictions_baseline = model.predict(X_test)

    # Converting labels to one_hot
    test_predictions_baseline = np.where(test_predictions_baseline >= 0.5, 1, 0)

    # OR each two image of every person
    pred_ls = []
    for i in range(0, test_predictions_baseline.shape[0] - 1, 2):
        temp = test_predictions_baseline[i] + test_predictions_baseline[i + 1]
        pred_ls.append(np.where(temp > 1, 1, temp))

    pred_shape = (
        test_predictions_baseline.shape[0] // 2,
        test_predictions_baseline.shape[1],
    )
    pred = np.stack(pred_ls, axis=0).reshape(pred_shape)

    print("shape of preds:", pred.shape)
    print("shape of Y_test:", Y_test.shape)

    # np.save(os.path.join(args.result_path, "pred_vgg16.npy"), pred)
    # np.save(os.path.join(args.result_path, "y_test_vgg16.npy"), Y_test)

    scores_dict = calc_metrics(Y_test, pred, threshold=0.5)
    pprint_metrics(scores_dict)

    p.plot_confusion_matrix_sns(
        Y_test,
        pred,
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )
    print(
        "Confusion Matrix saved in ",
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )


if __name__ == "__main__":
    main()
