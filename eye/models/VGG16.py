from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
import os
import sys

curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

from models.Model_base import ModelBase


class Vgg16(ModelBase):
    def __init__(self, input_shape, metrics, weights_path):
        super().__init__(input_shape, metrics, weights_path)

        self.model = models.Sequential()
        trainable = False
        # Block 1
        layer = layers.Conv2D(
            input_shape=self.input_shape,
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        self.model.add(layer)

        # Block 2
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        self.model.add(layer)

        # Block 3
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        self.model.add(layer)

        # Block 4
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        self.model.add(layer)

        # Block 5
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        self.model.add(layer)

        layer = layers.Flatten()
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = trainable
        self.model.add(layer)
        layer = layers.Dense(8, activation="sigmoid")
        self.model.add(layer)

    def image_net_load_weights(self):
        """
        This function loads the weights of our model from pretrained weightes of imagenet.
        """
        self.model.pop()
        layer = layers.Dense(1000, activation="softmax")
        layer.trainable = False
        self.model.add(layer)
        self.model.load_weights(self.weights_path)

        # Remove last layer
        self.model.pop()

        # Add new dense layer
        self.model.add(layers.Dense(8, activation="sigmoid"))

    def load_weights(self):
        """
        This function loads the weights of our model.
        """
        # Transfer learning, load previous weights
        self.model.load_weights(self.weights_path)

    def compile(self, loss="binary_crossentropy", lr=0.001):
        """
        This function compiles our model.

        Parameters
        ----------
        loss : str, optional
            type of loss function for compiling the model
        lr : float, optional
            learning rate for compiling the model

        Returns
        -------
        keras.Model
            The compiled model.
        """

        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        print("Configuration Start -------------------------")
        print(sgd.get_config())
        print("Configuration End -------------------------")
        self.model.compile(optimizer=sgd, loss=loss, metrics=self.metrics)

        self.show_summary(self.model)
        return self.model
