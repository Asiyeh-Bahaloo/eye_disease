from matplotlib.pyplot import axis
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from tensorflow.keras.losses import BinaryCrossentropy

from .model import KerasClsBaseModel
from keras import backend
from keras.engine import training
import h5py

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import layers
from keras.utils import data_utils

from .resnet_v2_imp import InceptionResNetV2
from .inception_v3_imp import InceptionV3
from .xception_imp import Xception
from .vgg16 import Vgg16


class XGBoost:
    def __init__(
        self,
        backbone,
        multi_label,
        # dropout_rate, add it later
        num_pop,
    ):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """

        self.backbone = backbone
        self.multi_label = multi_label
        self.num_pop = num_pop
        self.model_XGB = self.build(
            # dropout_rate=dropout_rate,
            multi_label=self.multi_label,
            num_pop=self.num_pop,
        )

    def build(self, multi_label, num_pop):

        XGB = XGBClassifier()
        if multi_label == "MultiOutputClassifier":
            model_XGB = MultiOutputClassifier(XGB)

        elif multi_label == "ClassifierChain":
            model_XGB = ClassifierChain(XGB, order="random", random_state=0)

        return model_XGB

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """

        self.backbone.load_imagenet_weights()

    def train(
        self,
        epochs,
        loss,
        metrics,
        callbacks=...,
        optimizer=...,
        num_pop=1,
        freeze_backbone=True,
        train_data_loader=None,
        validation_data_loader=None,
        X=None,
        Y=None,
        X_val=None,
        Y_val=None,
        batch_size=32,
        **kwargs,
    ):

        history = 0
        # Train backbone if intended
        if not freeze_backbone:
            history = self.backbone.train(
                epochs,
                loss,
                metrics,
                callbacks=callbacks,
                optimizer=optimizer,
                freeze_backbone=False,
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
                X=X,
                Y=Y,
                X_val=X_val,
                Y_val=Y_val,
                batch_size=batch_size,
                **kwargs,
            )

        # Pop the last layer
        model2 = Model(
            self.backbone.model.input, self.backbone.model.layers[-(num_pop + 1)].output
        )

        feat_list = []
        y_list = []
        feat_val_list = []
        y_val_list = []

        # get each batch from data loader
        for i in tqdm(range(len(train_data_loader))):
            x, y = train_data_loader.__iter__()
            feat = model2.predict(x)
            feat_list.append(feat)
            y_list.append(y)

        feat_from_cnn = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        for i in tqdm(range(len(validation_data_loader))):
            x, y = validation_data_loader.__iter__()
            feat = model2.predict(x)
            feat_val_list.append(feat)
            y_val_list.append(y)

        feat_val_from_cnn = np.concatenate(feat_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        ## Feed features extracted from backbone to XGBoost classifier
        self.model_XGB.fit(feat_from_cnn, y_train)

        pred_train_y = self.model_XGB.predict(feat_from_cnn)
        pred_val_y = self.model_XGB.predict(feat_val_from_cnn)

        training_result = {}
        validation_result = {}
        training_result_per_class = {}
        validation_result_per_class = {}

        print(f"y_pred shape= {pred_train_y.shape}")
        for i in range(10):
            name = metrics[i].__name__

            train_value = metrics[i](y_train, pred_train_y)
            training_result[name] = train_value

            validation_value = metrics[i](y_val, pred_val_y)
            validation_result[name] = validation_value

        tensor_y_train = tf.convert_to_tensor(y_train, np.float32)
        tensor_pred_train_y = tf.convert_to_tensor(pred_train_y, np.float32)
        tensor_y_val = tf.convert_to_tensor(y_val, np.float32)
        tensor_pred_val_y = tf.convert_to_tensor(pred_val_y, np.float32)

        # Per class metrics
        for i in range(10, len(metrics)):
            name = metrics[i].__name__

            train_value = metrics[i](tensor_y_train, tensor_pred_train_y)
            training_result_per_class[name] = train_value

            validation_value = metrics[i](tensor_y_val, tensor_pred_val_y)
            validation_result_per_class[name] = validation_value

        return (
            history,
            training_result,
            validation_result,
            training_result_per_class,
            validation_result_per_class,
        )

    def save(self, path, xgboost_path, **kwargs):
        """save function save model

        this function save svm layer and CNN net work

        Parameters
        ----------
        path : str
            path of file you wnat to save CNN model (.h5 file)
        xgboost_path : str
            path of file you want to save svm model (.pickle.dat file)
        """
        self.backbone.save(path, **kwargs)
        # save XGboost model to file
        pickle.dump(self.model_XGB, open(xgboost_path, "wb"))
