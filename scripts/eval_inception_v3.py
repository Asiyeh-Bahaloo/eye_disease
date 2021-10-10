import os
import sys
import argparse
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input,
)

from eye.models.inception_v3 import InceptionV3
from eye.utils.utils import load_data, pprint_metrics, calc_metrics
from eye.utils import plotter_utils as p
from eye.evaluation.metrics import (
    loss_per_class,
    accuracy_per_class,
    precision_per_class,
    recall_per_class,
    kappa_per_class,
    f1_per_class,
    auc_per_class,
    final_per_class,
    specificity_per_class,
    sensitivity_per_class,
)


def parse_arguments():
    # define arguments
    parser = argparse.ArgumentParser(
        description="Arguments for evaluating the inception_v3 model"
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        default="/Data/model_weights_inception_v3.h5",
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


# python eye/scripts/eval_inception_v3.py --weights=./Data/model_weights_inception_v3.h5 --data=./Data --result=./Data
def main():
    args = parse_arguments()
    tf.config.run_functions_eagerly(True)

    # Parameters
    num_classes = 8
    tag = "inception_v3"

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", tag)

    # Load data
    # TODO: use dataloaders instead
    (_, _), (_, _), (X_test, y_test) = load_data(args.data_folder)

    # TODO: Move preprocess to 'data' module and call them in dataloader
    X_test = preprocess_input(X_test)

    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    mlflow.log_param("Test data size", X_test.shape[0])

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    for l in range(num_classes):
        defined_metrics.append(loss_per_class(label=l))
        defined_metrics.append(accuracy_per_class(label=l))
        defined_metrics.append(precision_per_class(label=l))
        defined_metrics.append(recall_per_class(label=l))
        defined_metrics.append(kappa_per_class(label=l))
        defined_metrics.append(f1_per_class(label=l))
        defined_metrics.append(auc_per_class(label=l))
        defined_metrics.append(final_per_class(label=l))
        defined_metrics.append(specificity_per_class(label=l))
        defined_metrics.append(sensitivity_per_class(label=l))

    # Model
    model = InceptionV3(num_classes=num_classes)
    model.load_weights(path=args.weights_path)

    test_predictions_baseline = model.predict(X_test)

    metrics_name = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "auc",
    ]

    eval_metrics_name = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "kappa",
        "f1",
        "auc",
        "final",
        "specificity",
        "sensitivity",
    ]

    for name in class_names:
        for metric in eval_metrics_name:
            metrics_name.append(name + "_" + metric)

    baseline_results = model.evaluate(
        metrics=defined_metrics, loss=args.loss, X=X_test, Y=y_test
    )

    # prints detailed scores for each disease
    detailed_scores_dict = {
        name: score for score, name in zip(baseline_results, metrics_name)
    }
    for score in detailed_scores_dict.keys():
        mlflow.log_metric(score, detailed_scores_dict[score])

    pprint_metrics(detailed_scores_dict)

    # # prints only kappa, f1, auc, and final score
    # scores_dict = calc_metrics(Y_test, pred, threshold=0.5)
    # for score in scores_dict.keys():
    #     mlflow.log_param(score, scores_dict[score])
    # pprint_metrics(scores_dict)

    p.plot_confusion_matrix_sns(
        y_test,
        test_predictions_baseline,
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )
    print(
        "Confusion Matrix saved in ",
        os.path.join(args.result_path, f"{tag}_confusion_mat.png"),
    )

    mlflow.log_artifact(os.path.join(args.result_path, f"{tag}_confusion_mat.png"))
    mlflow.end_run()


if __name__ == "__main__":
    main()
