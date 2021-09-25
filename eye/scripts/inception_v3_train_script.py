
import sys
import os
sys.path.append(os.getcwd())

from eye.utils import data_loader_dataframe as data_loader
from eye.models import Inception_V3 as Inception

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam
import argparse as ap
from tensorflow.keras import metrics
from eye.train.inception_v3_train import train

# getting arguments from terminal
parser = ap.ArgumentParser(description="Train Inception model")
parser.add_argument(
    "dataframe", type=str, help="path of the .csv file that hold information about data"
)
parser.add_argument(
    "img_dir", type=str, help="path of the directory that contains the images"
)
parser.add_argument("batch_size", type=int, help="size of batchs")
parser.add_argument(
    "epochs", type=int, help="repetition of training through whole dataset"
)
args = parser.parse_args()


batch_size = args.batch_size
num_classes = 8
epochs = args.epochs
path_to_dataframe = args.dataframe
path_to_train_images = args.img_dir

model = Inception(8)
model.load_imagenet_weights()
model = model.base_model

# data loader using dataframe
data_loader = data_loader.data_loader(
    path_to_dataframe,
    path_to_train_images,
    bach_size=batch_size,
)

defined_metrics = [
    metrics.BinaryAccuracy(name="accuracy"),
    metrics.Precision(name="precision"),
    metrics.Recall(name="recall"),
    metrics.AUC(name="auc"),
]

model.compile(
    loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=defined_metrics
)

# adding early stopping to evoid overfitting
callback = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, mode="min", verbose=1
)

history, model = train(model, data_loader, batch_size, epochs)
# saving model in current directory
# print("saving")
# model.save(os.path.join("../model_weights", 'model_weights.h5'))
