import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input  # for preprocess

from eye.models.vgg19 import Vgg19
from eye.utils.utils import load_data, pprint_metrics, calc_metrics
from eye.utils import plotter_utils as p


def parse_arguments():
    # define arguments
    parser = argparse.ArgumentParser(
        description="Argumnts for training the VGG16 model"
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


# python eye/scripts/eval_vgg19.py --weights=./Data/model_weights_vgg19.h5 --data=./Data --result=./Data
def main():
    args = parse_arguments()

    # Parameters
    num_classes = 8
    tag = "vgg19"

    # Load data
    # TODO: use dataloaders instead
    (_, _), (_, _), (X_test, y_test) = load_data(args.data_folder)

    # TODO: Move preprocess to 'data' module and call them in dataloader
    X_test = preprocess_input(X_test)

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Model
    model = Vgg19(num_classes=num_classes, input_shape=(224, 224, 3))
    model.load_weights(path=args.weights_path)

    test_predictions_baseline = model.predict(X_test)

    baseline_results = model.evaluate(
        metrics=defined_metrics, loss=args.loss, X=X_test, Y=y_test
    )
    pprint_metrics(
        {
            name: score
            for score, name in zip(
                baseline_results, ["loss", "accuracy", "precision", "recall", "auc"]
            )
        }
    )

    scores_dict = calc_metrics(y_test, test_predictions_baseline, threshold=0.5)
    pprint_metrics(scores_dict)

    p.plot_confusion_matrix_sns(
        y_test,
        test_predictions_baseline,
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )
    print(
        "Confusion Matrix saved in ",
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )


if __name__ == "__main__":
    main()
