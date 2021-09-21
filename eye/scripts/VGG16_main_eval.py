import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.applications import vgg16

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)

from eye.evaluation.VGG16_eval import evaluat_vgg16
from eye.models.VGG16 import Vgg16
from eye.utils.utils import Plotter, load_data

# python eye/scripts/VGG16_main_eval.py --weights=/Data/model_weights_vgg16.h5 --data=/Data --result=/Data
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

    args = my_parser.parse_args()

    # Hyper parameters
    input_shape = (224, 224, 3)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.data_folder)
    X_test_drawing = X_test
    X_test = vgg16.preprocess_input(X_test)

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

    model.load_weights()
    model = model.compile()

    evaluat_vgg16(model=model, X_test=X_test, y_test=y_test)

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

    test_predictions_baseline = model.predict(X_test)
    plotter.plot_confusion_matrix_generic(
        y_test,
        test_predictions_baseline,
        os.path.join(args.result_path, "VGG16_confusionmat.png"),
    )

    # # plot output results
    # plotter.plot_output(
    #     test_predictions_baseline,
    #     y_test,
    #     X_test_drawing,
    #     os.path.join(args.result_path, "VGG16_output.png"),
    # )


if __name__ == "__main__":
    script_eval()
