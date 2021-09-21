import os
import sys
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)

from eye.models.VGG16 import Vgg16
from eye.train.VGG16_train import train_from_file
from eye.utils.utils import load_data, save_weights, Plotter
from tensorflow.keras.applications import vgg16

# python eye/scripts/VGG16_main_train.py --batch=2 --epoch=2 --weights=/Data/vgg16_weights_tf_dim_ordering_tf_kernels.h5 --data=/Data --result=/Data
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
        required=True,
    )
    my_parser.add_argument(
        "--epoch",
        dest="num_epochs",
        type=int,
        default=2,
        help="number of epochs you want to train your model",
        required=True,
    )

    my_parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        default="/Data/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
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
        dest="result",
        type=str,
        default="/Data",
        help="Path to the folder you want to save the model's results",
        required=True,
    )

    args = my_parser.parse_args()

    # import data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.data_folder)

    X_train = vgg16.preprocess_input(X_train)

    # Hyper parameters
    num_classes = 8
    input_shape = (224, 224, 3)

    # Metrics
    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model = Vgg16(
        input_shape=input_shape, metrics=defined_metrics, weights_path=args.weights_path
    )

    model.image_net_load_weights()
    model = model.compile()

    model, history = train_from_file(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=args.num_batch,
        epochs=args.num_epochs,
    )

    print("Saving models weights...")
    save_weights(model, os.path.join(args.result, "model_weights_vgg16.h5"))
    print("plotting...")

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

    plotter = Plotter(class_names)
    plotter.plot_accuracy(
        history=history, new_folder=os.path.join(args.result, "VGG16_accuracy.png")
    )


if __name__ == "__main__":
    script_train()
