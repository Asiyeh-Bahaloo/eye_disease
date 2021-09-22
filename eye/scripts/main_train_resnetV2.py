import sys, os

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)


from train.ResnetV2_training import resnet_v2_training
from utils.utils import load_data
from utils.ResnetV2_save_weights import save_weights
from utils.utils import Plotter
from models.ResnetV2 import Resnet_v2
import tensorflow as tf

import argparse


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
        "-a",
        "--add_weights",
        type=bool,
        help="add imagenet weights or not",
        required=True,
        default=True,
    )

    args = parser.parse_args()

    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)

    ### In initializing we have to pass image-net weights
    my_model = Resnet_v2((224, 224, 3), defined_metric, args.weight_path)

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data(args.data_path)

    if args.add_weights == True:
        my_model.load_image_net_weights(args.weight_path)

    my_model = my_model.compile()

    history = resnet_v2_training(
        x_train, y_train, x_val, y_val, my_model, BATCH_SIZE, EPOCHS
    )

    save_weights(my_model, args.result_path)
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

plotter = Plotter(class_names)
plotter.plot_metrics(history, os.path.join(args.result_path, "plot1.png"), 2)

print("plotting accuracy")
plotter.plot_accuracy(history, os.path.join(args.result_path, "plot2.png"))


############ For running
# python main_train_resnetV2.py -d /Dataset -w /Dataset/imagenet/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -r /Dataset -b 2 -e 2 -a True
