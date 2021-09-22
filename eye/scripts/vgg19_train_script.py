import os
import sys
import tensorflow as tf
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from train.vgg19_train import vgg19_train
from models.vgg19_model import Vgg19
from tensorflow.keras.applications import vgg19
from utils.utils import Plotter, load_data
import mlflow

# pyhton file.py data_folder_path weights.h5_path output_folder_path batch_size epochs patience loss
# python eye\scripts\vgg19_train_script.py --data_path ..\data --weights_path ..\weights\vgg19_weights.h5 --output_folder eye\train\vgg19_train_outputs --batch_size 8 --epoch 3 --patience 5 --loss binary_crossentropy --model_name vgg19_model.h5
# python eye/scripts/vgg19_train_script.py --data_path ../data --weights_path ../weights/vgg19_weights.h5 --output_folder eye/train/vgg19_train_outputs --batch_size 8 --epoch 3 --patience 5 --loss binary_crossentropy --model_name vgg19_model.h5

parser = argparse.ArgumentParser(description="Arguments for training the VGG19 model")
parser.add_argument("--data_folder", type=str, default=os.path.join("..", "data"))
parser.add_argument(
    "--weights_path",
    type=str,
    default=os.path.join("..", "weights", "vgg19_weights.h5"),
)
parser.add_argument(
    "--output_folder",
    type=str,
    default=os.path.join("eye", "train", "vgg19_train_outputs"),
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--loss", type=str, default="binary_crossentropy")
parser.add_argument("--model_name", type=str, default="vgg19_model.h5")
args = parser.parse_args()

mlflow.start_run()
mlflow.set_tag("mlflow.runName", args.model_name.split("_")[0])

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

print("defining metrics...")
defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

assert os.path.exists(args.weights_path), "Weights doesn't exist"

VGG19 = Vgg19((224, 224, 3), defined_metrics)
VGG19.load_imagenet_weights(args.weights_path)

lr = 1e-3
print("compiling model...")
VGG19.compile(loss=args.loss, lr=lr)

print("model summary...")
VGG19.model.summary()

assert os.path.exists(args.data_folder), "Data folder doesn't exist"

print("loading data...")
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(args.data_folder)

print("preprocessing data...")
x_train = vgg19.preprocess_input(x_train)
x_val = vgg19.preprocess_input(x_val)

mlflow.log_param("Batch size", args.batch_size)
mlflow.log_param("Epochs", args.epoch)
mlflow.log_param("Patience", args.patience)
mlflow.log_param("Loss", args.loss)
mlflow.log_param("Learning rate", lr)
mlflow.log_param("Training data size", x_train.shape[0])
mlflow.log_param("Validation data size", x_val.shape[0])
mlflow.log_param("Testing data size", x_test.shape[0])


VGG19.model, history = vgg19_train(
    VGG19.model,
    x_train,
    y_train,
    x_val,
    y_val,
    args.batch_size,
    args.epoch,
    args.patience,
)

print("saving weights")
model_file = os.path.join(args.output_folder, args.model_name)
VGG19.model.save(model_file)
mlflow.log_artifact(model_file)

tags = {"output_path": args.output_folder, "model_name": args.model_name}

# Set a batch of tags
mlflow.set_tags(tags)

print("Content of the model...")
baseline_results = VGG19.model.evaluate(x_test, y_test, verbose=2)
for name, value in zip(VGG19.model.metrics_names, baseline_results):
    print(name, ": ", value)


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
plotter = Plotter(class_names)

print("plotting metrics")
metric_plot_figure = os.path.join(args.output_folder, "metrics_plot.png")
plotter.plot_metrics(history, metric_plot_figure, 2)
mlflow.log_artifact(metric_plot_figure)

print("plotting accuracy")
accuracy_plot_figure = os.path.join(args.output_folder, "accuracy_plot.png")
plotter.plot_accuracy(history, accuracy_plot_figure)
mlflow.log_artifact(accuracy_plot_figure)

mlflow.end_run()
