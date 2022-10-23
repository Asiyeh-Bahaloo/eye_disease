import os, sys
import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

# from tensorflow.keras.models import Model
from keras.engine import training
from tensorflow.python.keras.utils import data_utils

from .model import KerasClsBaseModel

BASE_WEIGHT_URL = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/inception_resnet_v2/"
)


class InceptionResNetV2(KerasClsBaseModel):
    def __init__(
        self, num_classes, input_shape, dropout_rate=None, weight_decay_rate=None
    ):
        """Initialization of Resnet_v2 class which clarify model architecture"""

        super().__init__(num_classes, input_shape, dropout_rate, weight_decay_rate)

    def conv2d_bn(
        self,
        x,
        filters,
        kernel_size,
        dropout_rate=0.25,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=False,
        name=None,
    ):
        """Utility function to apply conv + BN.
        Args:
            x: input tensor.
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            strides: strides in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            use_bias: whether to use a bias in `Conv2D`.
            name: name of the ops; will become `name + '_ac'` for the activation
                and `name + '_bn'` for the batch norm layer.
        Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name,
        )(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x, training=True)
        if not use_bias:
            bn_axis = 1 if backend.image_data_format() == "channels_first" else 3
            bn_name = None if name is None else name + "_bn"
            x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + "_ac"
            x = layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(
        self, x, scale, block_type, block_idx, activation="relu", dropout_rate=None
    ):
        """Adds an Inception-ResNet block.
        This function builds 3 types of Inception-ResNet blocks mentioned
        in the paper, controlled by the `block_type` argument (which is the
        block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
        Args:
            x: input tensor.
            scale: scaling factor to scale the residuals (i.e., the output of passing
            `x` through an inception module) before adding them to the shortcut
            branch. Let `r` be the output from the residual branch, the output of this
            block will be `x + scale * r`.
            block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
            structure in the residual branch.
            block_idx: an `int` used for generating layer names. The Inception-ResNet
            blocks are repeated many times in this network. We use `block_idx` to
            identify each of the repetitions. For example, the first
            Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
            and the layer names will have a common prefix `'block35_0'`.
            activation: activation function to use at the end of the block (see
            [activations](../activations.md)). When `activation=None`, no activation
            is applied
            (i.e., "linear" activation: `a(x) = x`).
        Returns:
            Output tensor for the block.
        Raises:
            ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
        """
        if block_type == "block35":
            branch_0 = self.conv2d_bn(x, 32, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(x, 32, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(branch_1, 32, 3, dropout_rate=dropout_rate)
            branch_2 = self.conv2d_bn(x, 32, 1, dropout_rate=dropout_rate)
            branch_2 = self.conv2d_bn(branch_2, 48, 3, dropout_rate=dropout_rate)
            branch_2 = self.conv2d_bn(branch_2, 64, 3, dropout_rate=dropout_rate)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == "block17":
            branch_0 = self.conv2d_bn(x, 192, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(x, 128, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(branch_1, 160, [1, 7], dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(branch_1, 192, [7, 1], dropout_rate=dropout_rate)
            branches = [branch_0, branch_1]
        elif block_type == "block8":
            branch_0 = self.conv2d_bn(x, 192, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(x, 192, 1, dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(branch_1, 224, [1, 3], dropout_rate=dropout_rate)
            branch_1 = self.conv2d_bn(branch_1, 256, [3, 1], dropout_rate=dropout_rate)
            branches = [branch_0, branch_1]
        else:
            raise ValueError(
                "Unknown Inception-ResNet block type. "
                'Expects "block35", "block17" or "block8", '
                "but got: " + str(block_type)
            )

        block_name = block_type + "_" + str(block_idx)
        channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
        mixed = layers.Concatenate(axis=channel_axis, name=block_name + "_mixed")(
            branches
        )
        up = self.conv2d_bn(
            mixed,
            backend.int_shape(x)[channel_axis],
            1,
            dropout_rate=dropout_rate,
            activation=None,
            use_bias=True,
            name=block_name + "_conv",
        )

        x = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=backend.int_shape(x)[1:],
            arguments={"scale": scale},
            name=block_name,
        )([x, up])
        if activation is not None:
            x = layers.Activation(activation, name=block_name + "_ac")(x)
        return x

    def build(
        self,
        num_classes,
        input_shape,
        dropout_rate=None,
        weight_decay_rate=None,
    ):
        """builds the model architecture by default uses random weights

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        input_shape :

        Returns
        -------
        model [tensorflow.keras.models.Model]
              model's Architecture
        """

        img_input = layers.Input(shape=input_shape)

        x = self.conv2d_bn(
            img_input,
            32,
            3,
            dropout_rate=dropout_rate,
            strides=2,
            padding="valid",
        )
        x = self.conv2d_bn(x, 32, 3, dropout_rate=dropout_rate, padding="valid")
        x = self.conv2d_bn(x, 64, 3, dropout_rate=dropout_rate)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = self.conv2d_bn(x, 80, 1, dropout_rate=dropout_rate, padding="valid")
        x = self.conv2d_bn(x, 192, 3, dropout_rate=dropout_rate, padding="valid")
        x = layers.MaxPooling2D(3, strides=2)(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 320
        branch_0 = self.conv2d_bn(x, 96, 1, dropout_rate=dropout_rate)
        branch_1 = self.conv2d_bn(x, 48, 1, dropout_rate=dropout_rate)
        branch_1 = self.conv2d_bn(branch_1, 64, 5, dropout_rate=dropout_rate)
        branch_2 = self.conv2d_bn(x, 64, 1, dropout_rate=dropout_rate)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, dropout_rate=dropout_rate)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, dropout_rate=dropout_rate)
        branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, dropout_rate=dropout_rate)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
        x = layers.Concatenate(axis=channel_axis, name="mixed_5b")(branches)

        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = self.inception_resnet_block(
                x,
                scale=0.17,
                block_type="block35",
                block_idx=block_idx,
                dropout_rate=dropout_rate,
            )

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        branch_0 = self.conv2d_bn(
            x, 384, 3, dropout_rate=dropout_rate, strides=2, padding="valid"
        )
        branch_1 = self.conv2d_bn(x, 256, 1, dropout_rate=dropout_rate)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, dropout_rate=dropout_rate)
        branch_1 = self.conv2d_bn(
            branch_1,
            384,
            3,
            dropout_rate=dropout_rate,
            strides=2,
            padding="valid",
        )
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name="mixed_6a")(branches)

        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = self.inception_resnet_block(
                x,
                scale=0.1,
                block_type="block17",
                block_idx=block_idx,
                dropout_rate=dropout_rate,
            )

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = self.conv2d_bn(x, 256, 1, dropout_rate=dropout_rate)
        branch_0 = self.conv2d_bn(
            branch_0,
            384,
            3,
            dropout_rate=dropout_rate,
            strides=2,
            padding="valid",
        )
        branch_1 = self.conv2d_bn(x, 256, 1, dropout_rate=dropout_rate)
        branch_1 = self.conv2d_bn(
            branch_1,
            288,
            3,
            dropout_rate=dropout_rate,
            strides=2,
            padding="valid",
        )
        branch_2 = self.conv2d_bn(x, 256, 1, dropout_rate=dropout_rate)
        branch_2 = self.conv2d_bn(branch_2, 288, 3, dropout_rate=dropout_rate)
        branch_2 = self.conv2d_bn(
            branch_2,
            320,
            3,
            dropout_rate=dropout_rate,
            strides=2,
            padding="valid",
        )
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name="mixed_7a")(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = self.inception_resnet_block(
                x,
                scale=0.2,
                block_type="block8",
                block_idx=block_idx,
                dropout_rate=dropout_rate,
            )
        x = self.inception_resnet_block(
            x,
            scale=1.0,
            activation=None,
            block_type="block8",
            block_idx=10,
            dropout_rate=dropout_rate,
        )

        # Final convolution block: 8 x 8 x 1536
        x = self.conv2d_bn(x, 1536, 1, dropout_rate=dropout_rate, name="conv_7b")

        ## top part
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)

        inputs = img_input
        # Create model.
        model = training.Model(inputs, x, name="inception_resnet_v2")

        if weight_decay_rate is not None:
            l2_regularizer = regularizers.l2(weight_decay_rate)
            for layer in model.layers:
                if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                    model.add_loss(lambda: l2_regularizer(layer.kernel))
                if hasattr(layer, "bias_regularizer") and layer.use_bias:
                    model.add_loss(lambda: l2_regularizer(layer.bias))

        return model

    def load_imagenet_weights(self):
        """loads imagenet-pretrained weight into model"""

        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        fname = "inception_resnet_v2_weights_" "tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = data_utils.get_file(
            fname,
            BASE_WEIGHT_URL + fname,
            cache_subdir="models",
            file_hash="d19885ff4a710c122648d3b5c3b684e4",
        )
        self.model.load_weights(weights_path)

        x = self.model.layers[-1].output
        x = Dense(1024, activation="relu")(x)
        x = Dense(self.num_classes, activation="sigmoid")(x)
        self.model = training.Model(self.model.input, x)
