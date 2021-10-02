import os
import argparse

import mlflow
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input,
)

from eye.models.inception_v3 import InceptionV3
from eye.utils import plotter_utils as p
from eye.utils.utils import load_data, MlflowCallback


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="number of batchs you want to have in your training process",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs you want to train your model",
        required=False,
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=5,
        help="number of patience you want to use for early stopping",
        required=False,
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        default="binary_crossentropy",
        help="type of loss function with which you want to compile your model",
        required=False,
    )
    parser.add_argument(
        "--imgnetweights",
        dest="imagenet_weights_path",
        type=str,
        help="Path to the image net pretrained weights file",
        required=False,
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        help="Path to the model's pretrained weights file",
        required=False,
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
        dest="result",
        type=str,
        default="/Data",
        help="Path to the folder you want to save the model's results",
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        dest="lr",
        type=float,
        default=0.001,
        help="setting learning rate of optimizer",
    )
    parser.add_argument(
        "--decay_rate",
        dest="decay",
        type=float,
        default=1e-6,
        help="setting decay rate of optimizer",
    )
    parser.add_argument(
        "--momentum_rate",
        dest="momentum",
        type=float,
        default=0.9,
        help="setting momentum rate of optimizer",
    )
    parser.add_argument(
        "--nesterov_flag",
        dest="nesterov",
        type=bool,
        default=True,
        help="setting nesterov term of  optimizer: True or False",
    )

    args = parser.parse_args()
    return args


# python eye/scripts/train_inception_v3.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data
def main():
    args = parse_arguments()

    # Parameters
    num_classes = 8
    tag = "inception_v3"

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", tag)

    # Load data
    # TODO: use dataloaders instead
    (X_train, y_train), (X_val, y_val), (X_test, _) = load_data(args.data_folder)

    # TODO: Move preprocess to 'data' module and call them in dataloader
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)

    # Model
    model = InceptionV3(num_classes=num_classes)
    if args.imagenet_weights_path is not None:
        model.image_net_load_weights(weights_path=args.imagenet_weights_path)
    #need to check
    if args.weights_path is not None:
        model.load_weights(path=args.weights_path)

    mlflow.log_param("Batch size", args.batch_size)
    mlflow.log_param("Epochs", args.epochs)
    mlflow.log_param("Patience", args.patience)
    mlflow.log_param("Loss", args.loss)
    mlflow.log_param("Learning rate", args.lr)
    mlflow.log_param("Training data size", X_train.shape[0])
    mlflow.log_param("Validation data size", X_val.shape[0])
    mlflow.log_param("Testing data size", X_test.shape[0])

    # Optimizer
    # TODO: Define multiple optimizer
    sgd = SGD(
        lr=args.lr, decay=args.decay, momentum=args.momentum, nesterov=args.nesterov
    )

    # Metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Callbacks
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min", verbose=1
    )

    # Train
    history = model.train(
        epochs=args.epochs,
        loss=args.loss,
        metrics=metrics,
        callbacks=[MlflowCallback(), earlyStoppingCallback],
        optimizer=sgd,
        train_data_loader=None,
        validation_data_loader=None,
        X=X_train,
        Y=y_train,
        X_val=X_val,
        Y_val=y_val,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print("model trained successfuly.")

    # Save
    print("Saving models weights...")
    model_file = os.path.join(args.result, f"model_weights_{tag}.h5")
    model.save(model_file)
    mlflow.log_artifact(model_file)
    print(f"Saved model weights in {model_file}")

    # Set a batch of tags
    tags = {"output_path": args.result, "model_name": tag}
    mlflow.set_tags(tags)

    # Plot
    print("plotting...")
    accuracy_plot_figure = os.path.join(args.result, f"{tag}_accuracy.png")
    metrics_plot_figure = os.path.join(args.result, f"{tag}_metrics.png")
    p.plot_accuracy(history=history, path=accuracy_plot_figure)
    p.plot_metrics(history=history, path=metrics_plot_figure)

    mlflow.log_artifact(accuracy_plot_figure)
    mlflow.log_artifact(metrics_plot_figure)
    mlflow.end_run()


if __name__ == "__main__":
    main()
