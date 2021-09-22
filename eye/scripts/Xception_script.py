import sys
import os
import argparse
from tensorflow.keras.applications import xception
from numpy.lib.npyio import save
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from utils import utils
from models import Xception_model
from train import train_xception_model
from evaluation import Xception_evaluating


parser = argparse.ArgumentParser(
    description="for getting all of data and setting hyperparameters"
)
parser.add_argument(
    "--datapath",
    default=os.path.join("..", "data"),
)


parser.add_argument(
    "--num_classes", default=8, help="the number of classes of classifing task"
)

parser.add_argument(
    "--batch_size", default=3, help="the batch size of training process"
)
parser.add_argument("--epochs", default=1, help="setting epochs for training process")
parser.add_argument(
    "--patience", default=3, help="setting patience for callback finction"
)

parser.add_argument(
    "--lossFunction",
    default="binary_crossentropy",
    help="setting type of loss function",
)


parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="setting learning rate of optimizer",
)
parser.add_argument(
    "--decay_rate", type=float, default=1e-6, help="setting decay rate of optimizer"
)
parser.add_argument(
    "--momentum_rate",
    type=float,
    default=0.9,
    help="setting momentum rate of optimizer",
)
parser.add_argument(
    "--nesterov_flag",
    type=bool,
    default=True,
    help="setting nesterov term of  optimizer: True or False",
)

parser.add_argument(
    "--weight_url",
    type=str,
    default=r"eye-disease\eye\scripts\xception_model_weights.h5",
    help="setting the url of pre trained weights file ",
)

parser.add_argument(
    "--folder",
    type=str,
    default=r"eye-disease\eye\scripts\plot_4_metrics",
    help="setting the url and name of .png file of plot metrics diagram",
)


parser.add_argument(
    "--model_path",
    type=str,
    default=r"eye-disease\eye\scripts\xception_model.h5",
    help="setting the url and name of .h5 file that we want to save model there",
)

args = parser.parse_args()

(x_train, y_train), (x_val, y_val), (x_test, y_test) = utils.load_data(args.datapath)

x_train = xception.preprocess_input(x_train)
x_val = xception.preprocess_input(x_val)
x_test = xception.preprocess_input(x_test)


new_model = Xception_model.xceptionModel(num_classes=args.num_classes)
new_model.load_xception_weight(args.weight_url)
model = new_model.base_model


sgd = SGD(
    lr=args.learning_rate,
    decay=args.decay_rate,
    momentum=args.momentum_rate,
    nesterov=args.nesterov_flag,
)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]


trained_model, his = train_xception_model.training_model(
    model,
    defined_metrics,
    sgd,
    args.lossFunction,
    x_train,
    y_train,
    x_val,
    y_val,
    args.epochs,
    args.batch_size,
    True,
    args.patience,
)


# model.save(args.model_path)

t, p = Xception_evaluating.predict_xception_model(
    x_test=x_test, y_test=y_test, model=trained_model
)

print(t)
print(p)


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
plotter = utils.Plotter(class_names)
print("plotting metrics")
plotter.plot_metrics(his, args.folder, 2)
