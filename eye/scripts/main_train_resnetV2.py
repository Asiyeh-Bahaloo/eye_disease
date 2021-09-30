import os
import argparse

import mlflow
import tensorflow as tf

from eye.models.ResnetV2 import Resnet_v2
from eye.train.ResnetV2_training import resnet_v2_training
from eye.utils.utils import load_data
from eye.utils import plotter_utils as p


if __name__ == "__main__":
    defined_metric = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="the address of your data",
        required=True,
        default="/Dataset",
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        type=str,
        help="Path to imagenet weights",
        required=True,
        default="/Dataset",
    )

    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        help="where to save to weights?",
        required=True,
        default="/Dataset",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="size of the batch",
        required=False,
        default=4,
    )

    parser.add_argument(
        "-e", "--epochs", type=int, help="number of epochs", required=True, default=2
    )

    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        help="number of patience for early stopping",
        required=True,
        default=5,
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        help="type of loss function",
        required=True,
        default="binary_crossentropy",
    )

    parser.add_argument(
        "-a",
        "--add_weights",
        type=bool,
        help="add imagenet weights or not",
        required=True,
        default=True,
    )

    args = parser.parse_args()

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", "resnetV2")

    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    PATIENCE = int(args.patience)

    ### In initializing we have to pass image-net weights
    my_model = Resnet_v2((224, 224, 3), defined_metric, args.weight_path)

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data(args.data_path)

    if args.add_weights == True:
        my_model.load_image_net_weights(args.weight_path)

    lr = 0.001
    my_model = my_model.compile(args.loss, lr)

    mlflow.log_param("Batch size", args.batch_size)
    mlflow.log_param("Epochs", args.epochs)
    mlflow.log_param("Patience", args.patience)
    mlflow.log_param("Loss", args.loss)
    mlflow.log_param("Learning rate", lr)
    mlflow.log_param("Training data size", x_train.shape[0])
    mlflow.log_param("Validation data size", x_val.shape[0])
    mlflow.log_param("Testing data size", x_test.shape[0])

    history = resnet_v2_training(
        x_train, y_train, x_val, y_val, my_model, BATCH_SIZE, EPOCHS, PATIENCE
    )

    print("saving weights")
    model_file = os.path.join(args.result_path, "model_weights_resnetv2.h5")
    my_model.save(model_file)
    # save_weights(my_model, args.result_path)
    mlflow.log_artifact(model_file)

    tags = {"output_path": args.result_path, "model_name": "resnetV2"}

    # Set a batch of tags
    mlflow.set_tags(tags)

    print("plotting metrics", args.result_path)
    class_names = [
        "Normal",
        "Diabetes",
        "Glaucoma",
        "Cataract",
        "AMD",
        "Hypertension",
        "Myopia",
        "Others",
    ]

print("plotting metrics")
metric_plot_figure = os.path.join(args.result_path, "metrics_plot.png")
p.plot_metrics(history, os.path.join(args.result_path, "metrics_plot.png"))
mlflow.log_artifact(metric_plot_figure)

print("plotting accuracy")
accuracy_plot_figure = os.path.join(args.result_path, "accuracy_plot.png")
p.plot_accuracy(history, os.path.join(args.result_path, "accuracy_plot.png"))
mlflow.log_artifact(accuracy_plot_figure)

mlflow.end_run()

############ For running
# python main_train_resnetV2.py -d /Dataset -w /Dataset/imagenet/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -r /Dataset -b 2 -e 2 -p 5 -l binary_crossentropy -a True
