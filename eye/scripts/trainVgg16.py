import os
import sys
import argparse
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import vgg16  # for preprocess

from eye.models.vgg16 import Vgg16
from eye.utils import plotter_utils as p
from eye.utils.utils import load_data, MlflowCallback

# python eye/scripts/trainVgg16.py --batch=2 --epoch=2 --patience=5 --loss=binary_crossentropy --weights=./Data/model_weights_vgg16.h5 --data=./Data --result=./Data
def script_train():
    my_parser = argparse.ArgumentParser(
        description="Argumnts for training the VGG16 model"
    )
    my_parser.add_argument(
        "--batch",
        dest="num_batch",
        type=int,
        default=2,
        help="number of batchs you want to have in your training process",
        required=False,
    )
    my_parser.add_argument(
        "--epoch",
        dest="num_epochs",
        type=int,
        default=2,
        help="number of epochs you want to train your model",
        required=False,
    )
    my_parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=5,
        help="number of patience you want to use for early stopping",
        required=False,
    )
    my_parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        default="binary_crossentropy",
        help="type of loss function with which you want to compile your model",
        required=False,
    )
    my_parser.add_argument(
        "--imgnetweights",
        dest="imagenet_weights_path",
        type=str,
        help="Path to the image net pretrained weights file",
        required=False,
    )
    my_parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        help="Path to the model's pretrained weights file",
        required=False,
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
        dest="result",
        type=str,
        default="/Data",
        help="Path to the folder you want to save the model's results",
        required=True,
    )

    args = my_parser.parse_args()

    # Parameters
    num_classes = 8
    input_shape = (224, 224, 3)
    lr = 0.001

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", "vgg16")

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.data_folder)

    X_train = vgg16.preprocess_input(X_train)

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Model
    model = Vgg16(num_class=num_classes, input_shape=input_shape)
    if args.imagenet_weights_path is not None:
        model.image_net_load_weights(weights_path=args.weights_path)

    if args.weights_path is not None:
        model.load_weights(path=args.weights_path)

    mlflow.log_param("Batch size", args.num_batch)
    mlflow.log_param("Epochs", args.num_epochs)
    mlflow.log_param("Patience", args.patience)
    mlflow.log_param("Loss", args.loss)
    mlflow.log_param("Learning rate", lr)
    mlflow.log_param("Training data size", X_train.shape[0])
    mlflow.log_param("Validation data size", X_val.shape[0])
    mlflow.log_param("Testing data size", X_test.shape[0])

    # Optimizer
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    # Callbacks
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min", verbose=1
    )

    # Train
    history = model.train(
        epochs=args.num_epochs,
        loss="binary_crossentropy",
        metrics=defined_metrics,
        callbacks=[MlflowCallback(), earlyStoppingCallback],
        optimizer=sgd,
        train_data_loader=None,
        validation_data_loader=None,
        X=X_train,
        Y=y_train,
        X_val=X_val,
        Y_val=y_val,
        batch_size=args.num_batch,
        shuffle=True,
    )
    print("model trained successfuly.")

    # Save
    print("Saving models weights...")
    model_file = os.path.join(args.result, "model_weights_vgg16.h5")
    model.save(model_file)
    mlflow.log_artifact(model_file)

    # Set a batch of tags
    tags = {"output_path": args.result, "model_name": "vgg16"}
    mlflow.set_tags(tags)

    # Plot
    print("plotting...")
    accuracy_plot_figure = os.path.join(args.result, "VGG16_accuracy.png")
    metrics_plot_figure = os.path.join(args.result, "VGG16_metrics.png")
    p.plot_accuracy(history=history, path=accuracy_plot_figure)
    p.plot_metrics(history=history, path=metrics_plot_figure)

    mlflow.log_artifact(accuracy_plot_figure)
    mlflow.log_artifact(metrics_plot_figure)
    mlflow.end_run()


if __name__ == "__main__":
    script_train()
