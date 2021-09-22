import sys, os

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from evaluation.ResnetV2_evaluation import resnet_v2_evaluate
from utils.utils import load_data
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
        help="Path to the previously saved weights",
        required=True,
        default="/Dataset",
    )

    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        help="Where to save results such as final_score?",
        required=True,
        default="/Dataset",
    )

    args = parser.parse_args()

    my_model = Resnet_v2((224, 224, 3), defined_metric, args.weight_path)

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data(args.data_path)

    ### Using imagenet weights
    my_model.load_image_net_weights(args.weight_path)

    ## Using our previous weights which we trained
    # my_model.load_weightss(args.weight_path)

    # my_model.costomizeModel()
    my_model = my_model.compile()

    resnet_v2_evaluate(x_test, y_test, my_model, args.result_path)


### running
# python main_evaluation_resnetV2.py -d /Dataset -w /Dataset/imagenet/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -r /Dataset
# python main_evaluation_resnetV2.py -d /Dataset -w /Dataset/model_weights_resnetv2.h5 -r /Dataset
