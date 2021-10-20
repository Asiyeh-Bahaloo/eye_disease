import os
import argparse
import mlflow
import tensorflow as tf
import glob
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

from eye.models.vgg19 import Vgg19
from eye.utils.utils import pprint_metrics, calc_metrics
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
    specificity,
)
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
        description="Arguments for evaluating the VGG19 model"
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        default="/Data/model_weights_vgg19.h5",
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


# python scripts/eval_vgg19.py --weights=./Data/model_weights_vgg19.h5 --data=./Data --result=./Data
def main():
    args = parse_arguments()
    tf.config.run_functions_eagerly(True)

    # Parameters
    num_classes = 8
    tag = "vgg19"

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", tag)

    # Load data
    compose_test = Compose(
        transforms=[
            RemovePadding(),
            # BenGraham(350),
            Resize((224, 224), False),
            KerasPreprocess(model_name="vgg19"),
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
    Y_test2_ls = []

    for idx, filename in enumerate(tqdm(sorted(glob.glob("*.jpg")))):

        # handling image
        img = cv2.imread(filename)
        img = compose_test(img)
        X_test_ls.append(img)

        # handling label
        if idx % 2 == 0:
            image_id = int(filename[: filename.find("_")])
            Y_test_ls.append(df.loc[df["ID"] == image_id, class_names].to_numpy())

        Y_test2_ls.append(Y_test_ls[-1])

    # change directory from Data to main directory
    os.chdir("/usr/src/app")

    img_shape = (len(X_test_ls),) + (X_test_ls[0].shape)
    label_shape = (len(Y_test_ls),) + (Y_test_ls[0].shape[1],)
    label2_shape = (len(Y_test2_ls),) + (Y_test2_ls[0].shape[1],)

    X_test = np.stack(X_test_ls, axis=0).reshape(img_shape)
    Y_test = np.stack(Y_test_ls, axis=0).reshape(label_shape)
    Y_test2 = np.stack(Y_test2_ls, axis=0).reshape(label2_shape)

    mlflow.log_param("Test data size", X_test.shape[0])

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        specificity,
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
        # defined_metrics.append(sensitivity_per_class(label=l))

    # Model
    model = Vgg19(num_classes=num_classes, input_shape=(224, 224, 3))
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
        metrics=defined_metrics, loss=args.loss, X=X_test, Y=Y_test2
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
        Y_test,
        pred,
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
