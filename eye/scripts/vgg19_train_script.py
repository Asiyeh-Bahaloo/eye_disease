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
from tensorflow.keras.optimizers import SGD
from utils.utils import Plotter, load_data

# pyhton file.py data_folder_path weights.h5_path output_folder_path batch_size epochs patience loss
# python eye\scripts\vgg19_train_script.py --data ..\data --weights ..\weights\vgg19_weights_tf_dim_ordering_tf_kernels.h5 --output eye\train\vgg19_train_outputs --batch_size 8 --epoch 3 --patience 5 --loss binary_crossentropy
# python eye/scripts/vgg19_train_script.py --data ../data --weights ../weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5 --output eye/train/vgg19_train_outputs --batch_size 8 --epoch 3 --patience 5 --loss binary_crossentropy

parser = argparse.ArgumentParser(description="Arguments for training the VGG19 model")
parser.add_argument("--data", type=str, default=os.path.join("..", "data"))
parser.add_argument(
    "--weights",
    type=str,
    default=os.path.join(
        "..", "weights", "vgg19_weights_tf_dim_ordering_tf_kernels.h5"
    ),
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join("eye", "train", "vgg19_train_outputs"),
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--loss", type=str, default="binary_crossentropy")
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

print("defining metrics...")
defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

assert os.path.exists(args.weights), "Weights doesn't exist"

VGG19 = Vgg19((224, 224, 3), defined_metrics, "ImageNet", args.weights)

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=False)
print("compiling model...")
VGG19.model.compile(optimizer=sgd, loss=args.loss, metrics=defined_metrics)

print("model summary...")
VGG19.model.summary()

assert os.path.exists(args.data), "Data folder doesn't exist"

print("loading data...")
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(args.data)

print("preprocessing data...")
x_train = vgg19.preprocess_input(x_train)
x_val = vgg19.preprocess_input(x_val)

VGG19.model, history = vgg19_train(
    VGG19.model, x_train, y_train, x_val, y_val, batch_size=8, epochs=3, patience=5
)

print("saving weights")
VGG19.model.save(os.path.join(args.output, "vgg19_model.h5"))

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
plotter.plot_metrics(history, os.path.join(args.output, "plot1.png"), 2)

print("plotting accuracy")
plotter.plot_accuracy(history, os.path.join(args.output, "plot2.png"))
