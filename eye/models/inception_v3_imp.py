import tensorflow as tf
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import layers
from keras.utils import data_utils
from keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export

from .model import KerasClsBaseModel

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


class InceptionV3(KerasClsBaseModel):
    def __init__(self, num_classes, input_shape):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """
        # super().__init__(num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.build(self.num_classes, self.input_shape)

    def conv2d_bn(
        self,
        x,
        filters,
        num_row,
        num_col,
        padding="same",
        strides=(1, 1),
        name=None,
        dropout_rate=None,
    ):
        """Utility function to apply conv + BN.
        Args:
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        # if backend.image_data_format() == "channels_first":
        #     bn_axis = 1
        # else:
        #     bn_axis = 3
        bn_axis = 3
        x = layers.Conv2D(
            filters,
            (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        )(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = layers.Activation("relu", name=name)(x)
        return x

    def build(self, num_classes, input_shape):

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
        channel_axis = 3  ## channels_last data format

        x = self.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding="valid")
        x = self.conv2d_bn(x, 32, 3, 3, padding="valid")
        x = self.conv2d_bn(x, 64, 3, 3)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding="valid")
        x = self.conv2d_bn(x, 192, 3, 3, padding="valid")
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed0",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed1",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 2: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed2",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding="valid")

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding="valid"
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed3")(
            [branch3x3, branch3x3dbl, branch_pool]
        )

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed4",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(5 + i),
            )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=0.25)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed7",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch3x3 = self.conv2d_bn(
            branch3x3, 320, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed8")(
            [branch3x3, branch7x7x3, branch_pool]
        )

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1, dropout_rate=0.25)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3, dropout_rate=0.25)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.Concatenate(axis=channel_axis, name="mixed9_" + str(i))(
                [branch3x3_1, branch3x3_2]
            )

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1, dropout_rate=0.25)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3, dropout_rate=0.25)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.Concatenate(axis=channel_axis)(
                [branch3x3dbl_1, branch3x3dbl_2]
            )

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(9 + i),
            )([branch1x1, branch3x3, branch3x3dbl, branch_pool])

        inputs = img_input
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)

        model = training.Model(inputs, x, name="inception_v3")
        return model

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """

        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        weights_path = data_utils.get_file(
            "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="bcbd6486424b2319ff4ef7d526e38f63",
        )
        self.model.load_weights(weights_path)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(self.num_classes, activation="sigmoid")(x)

        self.model = training.Model(self.model.input, x)
