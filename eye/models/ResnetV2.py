import os, sys
import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# global backend, layers, models, keras_utils
# backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

# import tensorflow.keras_utils


from abc import abstractmethod

num_classes = 8

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from models.Model_base import ModelBase


class Resnet_v2(ModelBase):
    """[Our Resnt_v2 class]

    Parameters
    ----------
    ModelBase : [class]
        [Inheritance from ModelBase class]
    """

    def __init__(self, input_shape, metrics, weights_path):

        """[Initialization of Resnet_v2 class which clarify model architecture]"""

        super().__init__(input_shape, metrics, weights_path)

        self.model = inception_resnet_v2.InceptionResNetV2(
            weights=None, include_top=False
        )
        base_model = self.model

        self.model = self.model.output
        self.model = GlobalAveragePooling2D()(self.model)
        self.model = Dense(1024, activation="relu")(self.model)
        self.model = Dense(num_classes, activation="sigmoid")(self.model)
        self.model = Model(inputs=base_model.input, outputs=self.model)

    def compile(self, loss="binary_crossentropy", lr=0.001):
        """[This functin will compile our model]
        Parameters
        ----------
        loss : str
            type of loss function for compiling the model
        lr : float
            learning rate for compiling the model
        Returns
        -------
        [A compiled Model]

        """
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=False)
        print("Configuration Start -------------------------")
        print(sgd.get_config())
        print("Configuration End -------------------------")
        self.model.compile(optimizer=sgd, loss=loss, metrics=self.metrics)

        return self.model

    def load_weightss(self, weights_path):
        """load_weight function used for loading pretrained weights

        These types of weights are perfectly tailored to the personalized model

        Parameters
        ----------
         weights_path : str
            it is the address of pretrained weights that we use
        """
        self.model.load_weights(weights_path)

        # self.model = self.model(weights=weights_path, include_top=False)

    def load_image_net_weights(self, weights_path):

        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part

        Parameters
        ----------
        weights_path : str
            it is the address of pretrained weights that we use
        """

        self.model = inception_resnet_v2.InceptionResNetV2(
            weights="imagenet", include_top=False
        )
        base_model = self.model

        self.model = self.model.output
        self.model = GlobalAveragePooling2D()(self.model)
        self.model = Dense(1024, activation="relu")(self.model)
        self.model = Dense(num_classes, activation="sigmoid")(self.model)
        self.model = Model(inputs=base_model.input, outputs=self.model)
