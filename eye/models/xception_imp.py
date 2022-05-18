import tensorflow as tf

# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
import tensorflow.compat.v2 as tf
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export

from .model import KerasClsBaseModel

TF_WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


class Xception(KerasClsBaseModel):
    """Xception a model based on Xception model to detect disease

    Parameters
    ----------
    num_classes : int
        number of classes of the classification task
    """

    def __init__(self, num_classes, input_shape, dropout_rate=None):
        """__init__ set number of classes and builds model architecture

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        """
        # super().__init__(num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.model = self.build(self.num_classes, self.input_shape, self.dropout_rate)

    def build(self, num_classes, input_shape, dropout_rate=None):
        """builds the model architecture by default uses random weights

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        pretrained_backbone : tensorflow.keras.models.Model, optional
            contains backbone model with pretrained weights, by default None

        Returns
        -------
        model [tensorflow.keras.models.Model]
              model's Architecture
        """

        img_input = layers.Input(shape=input_shape)
        channel_axis = -1  ##channels_last data format

        x = layers.Conv2D(
            32, (3, 3), strides=(2, 2), use_bias=False, name="block1_conv1"
        )(img_input)
        x = layers.BatchNormalization(axis=channel_axis, name="block1_conv1_bn")(x)
        x = layers.Activation("relu", name="block1_conv1_act")(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name="block1_conv2")(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block1_conv2_bn")(x)
        x = layers.Activation("relu", name="block1_conv2_act")(x)

        residual = layers.Conv2D(
            128, (1, 1), strides=(2, 2), padding="same", use_bias=False
        )(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(
            128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv1_bn")(x)
        x = layers.Activation("relu", name="block2_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv2_bn")(x)

        x = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), padding="same", name="block2_pool"
        )(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(
            256, (1, 1), strides=(2, 2), padding="same", use_bias=False
        )(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation("relu", name="block3_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv1_bn")(x)
        x = layers.Activation("relu", name="block3_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv2_bn")(x)

        x = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), padding="same", name="block3_pool"
        )(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(
            728, (1, 1), strides=(2, 2), padding="same", use_bias=False
        )(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation("relu", name="block4_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv1_bn")(x)
        x = layers.Activation("relu", name="block4_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv2_bn")(x)

        x = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), padding="same", name="block4_pool"
        )(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = "block" + str(i + 5)

            x = layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
            x = layers.SeparableConv2D(
                728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv1"
            )(x)
            if (
                dropout_rate is not None and i == 7
            ):  ## put drop just for the last iteration in loop , because we want 8 dropouts totaaly: 3+5
                x = layers.Dropout(dropout_rate)(x, training=True)
            x = layers.BatchNormalization(
                axis=channel_axis, name=prefix + "_sepconv1_bn"
            )(x)
            x = layers.Activation("relu", name=prefix + "_sepconv2_act")(x)
            x = layers.SeparableConv2D(
                728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv2"
            )(x)
            if dropout_rate is not None and i == 7:
                x = layers.Dropout(dropout_rate)(x, training=True)
            x = layers.BatchNormalization(
                axis=channel_axis, name=prefix + "_sepconv2_bn"
            )(x)
            x = layers.Activation("relu", name=prefix + "_sepconv3_act")(x)
            x = layers.SeparableConv2D(
                728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv3"
            )(x)
            if dropout_rate is not None and i == 7:
                x = layers.Dropout(dropout_rate)(x, training=True)
            x = layers.BatchNormalization(
                axis=channel_axis, name=prefix + "_sepconv3_bn"
            )(x)

            x = layers.add([x, residual])

        residual = layers.Conv2D(
            1024, (1, 1), strides=(2, 2), padding="same", use_bias=False
        )(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x, training=True)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation("relu", name="block13_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1"
        )(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=channel_axis, name="block13_sepconv1_bn")(x)
        x = layers.Activation("relu", name="block13_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2"
        )(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=channel_axis, name="block13_sepconv2_bn")(x)

        x = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), padding="same", name="block13_pool"
        )(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(
            1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1"
        )(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=channel_axis, name="block14_sepconv1_bn")(x)
        x = layers.Activation("relu", name="block14_sepconv1_act")(x)

        x = layers.SeparableConv2D(
            2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv2"
        )(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=channel_axis, name="block14_sepconv2_bn")(x)
        x = layers.Activation("relu", name="block14_sepconv2_act")(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(num_classes, activation="sigmoid")(x)

        inputs = img_input
        model = training.Model(inputs, x, name="xception")

        return model

    def load_imagenet_weights(self):
        """loads imagenet-pretrained weight into model"""

        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        weights_path = data_utils.get_file(
            "xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="b0042744bf5b25fce3cb969f33bebb97",
        )
        self.model.load_weights(weights_path)

        x = self.model.layers[-1].output
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(self.num_classes, activation="sigmoid")(x)

        self.model = training.Model(self.model.input, x)
