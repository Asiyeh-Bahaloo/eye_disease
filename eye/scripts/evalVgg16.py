import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16

from eye.models.vgg16 import Vgg16
from eye.utils.utils import print_metrics, load_data
from eye.utils import plotter_utils as p

# python eye/scripts/evalVgg16.py --weights=./Data/model_weights_vgg16.h5 --data=./Data --result=./Data
def script_eval():

    # define arguments
    my_parser = argparse.ArgumentParser(
        description="Argumnts for training the VGG16 model"
    )
    my_parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        default="/Data/model_weights_vgg16.h5",
        help="Path to the model's weights file",
        required=True,
    )
    my_parser.add_argument(
        "--data",
        dest="data_folder",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )
    my_parser.add_argument(
        "--result",
        dest="result_path",
        type=str,
        default="/Data",
        help="Path to the folder you want to save your results",
        required=True,
    )
    my_parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        default="binary_crossentropy",
        help="type of loss function with which you want to compile your model",
        required=False,
    )

    args = my_parser.parse_args()

    # Parameters
    input_shape = (224, 224, 3)
    num_classes = 8

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.data_folder)

    X_test = vgg16.preprocess_input(X_test)

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Model
    model = Vgg16(num_class=num_classes, input_shape=input_shape)

    if args.weights_path is not None:
        model.load_weights(path=args.weights_path)

    test_predictions_baseline = model.predict(X_test)
    baseline_results = model.evaluate(
        metrics=defined_metrics, loss=args.loss, X=X_test, Y=y_test
    )
    print("baseline_results", baseline_results)

    print_metrics(y_test, test_predictions_baseline, threshold=0.5)

    test_predictions_baseline = model.predict(X_test)

    p.plot_confusion_matrix_sns(
        y_test,
        test_predictions_baseline,
        os.path.join(args.result_path, "VGG16_confusionmat.png"),
    )
    print(
        "Confusion Matrix saved in ",
        os.path.join(args.result_path, "VGG16_confusionmat.png"),
    )


if __name__ == "__main__":
    script_eval()
