import os
import sys
import os
sys.path.append(os.getcwd())

from tensorflow.keras.losses import binary_crossentropy
import eye.utils.data_loader_dataframe as data_loader
from eye.models.Inception_V3 import Inception_V3
import argparse as ap
from eye.evaluation import Inception_V3_evaluate

#getting arguments through argparser
parser = ap.ArgumentParser(description='evaluation of inception model')
parser.add_argument('model', type=str, help='path to model.h5 that will be evaluated')
parser.add_argument('dataframe', type=str, help='path to .csv file')
parser.add_argument('img_dir', type=str, help='path to images directory')
parser.add_argument('batch_size', type=int, help='size of batchs')

args = parser.parse_args()


batch_size = args.batch_size
path_to_datafram = args.dataframe
path_to_images = args.img_dir
path_to_model = args.model

data_loader = data_loader.data_loader(path_to_datafram, path_to_images, batch_size)

model = Inception_V3(num_calsses=8)
model.load_inception_weight(path_to_model)
model.compile(loss= binary_crossentropy)

Inception_V3_evaluate.eval(model.base_model, data_loader, batch_size)
