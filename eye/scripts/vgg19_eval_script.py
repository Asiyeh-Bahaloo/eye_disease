import os
import sys
import argparse

import tensorflow as tf
from tensorflow.keras.applications import vgg19

from eye.utils.utils import load_data, save_predict_output, print_metrics
from eye.utils import plotter_utils as p
from eye.evaluation.vgg19_evaluation import vgg19_evaluation

# pyhton file.py data_folder_path output_folder_path model_path
# python eye\scripts\vgg19_eval_script.py --data ..\data --output eye\evaluation\vgg19_evaluation_outputs --model eye\train\vgg19_train_outputs\vgg19_model.h5
# python eye/scripts/vgg19_eval_script.py --data ../data --output eye/evaluation/vgg19_evaluation_outputs --model eye/train/vgg19_train_outputs/vgg19_model.h5


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=os.path.join("..", "data"))
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join("eye", "evaluation", "vgg19_evaluation_outputs"),
)
parser.add_argument(
    "--model",
    type=str,
    default=os.path.join("eye", "train", "vgg19_train_outputs", "vgg19_model.h5"),
)
args = parser.parse_args()

assert os.path.exists(args.data), "Data folder doesn't exist"

print("loading data...")
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(args.data)

x_test_drawing = x_test

print("preprocessing data...")
x_test = vgg19.preprocess_input(x_test)

if not os.path.exists(args.output):
    os.makedirs(args.output)

assert os.path.exists(args.model), "Model is not saved"

print("loading model...")
model = tf.keras.models.load_model(args.model)

y_pred = vgg19_evaluation(model, x_test)

print("saveing the predictions...")
save_predict_output(y_pred, os.path.join(args.output, "VGG19_prediction.csv"))

print("printing the final score...")
print_metrics(y_test, y_pred, threshold=0.5)

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

# plot data input


print("plotting confusion matrix")
p.plot_confusion_matrix_sns(y_test, y_pred, os.path.join(args.output, "plot3.png"))

print("plottig output results...")
p.plot_output(
    y_pred, y_test, x_test_drawing, os.path.join(args.output, "plot4.png"), class_names
)
