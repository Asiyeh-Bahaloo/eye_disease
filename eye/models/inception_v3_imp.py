import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras.engine import training
import h5py
import numpy as np

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import data_utils


from .model import KerasClsBaseModel

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


def load_attributes_from_hdf5_group(group, name):

    """Loads attributes of the specified name from the HDF5 group.
    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.
    Returns:
        data: Attributes data.
    """
    if name in group.attrs:
        data = [
            n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]
        ]
    else:
        data = []
        chunk_id = 0
        while "%s%d" % (name, chunk_id) in group.attrs:
            data.extend(
                [
                    n.decode("utf8") if hasattr(n, "decode") else n
                    for n in group.attrs["%s%d" % (name, chunk_id)]
                ]
            )
            chunk_id += 1
    return data


def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.
    Args:
        f: A pointer to a HDF5 group.
    Returns:
        List of NumPy arrays of the weight values.
    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")
    return [np.asarray(f[weight_name]) for weight_name in weight_names]


class InceptionV3(KerasClsBaseModel):
    def __init__(
        self, num_classes, input_shape, dropout_rate=None, weight_decay_rate=None
    ):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """
        super().__init__(num_classes, input_shape, dropout_rate, weight_decay_rate)

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
        pretrained_backbone : tensorflow.keras.models.Model, optional
            contains backbone model with pretrained weights, by default None

        Returns
        -------
        model [tensorflow.keras.models.Model]
              model's Architecture
        """

        img_input = layers.Input(shape=input_shape)
        channel_axis = 3  ## channels_last data format

        x = self.conv2d_bn(
            img_input,
            32,
            3,
            3,
            strides=(2, 2),
            padding="valid",
            dropout_rate=dropout_rate,
        )
        x = self.conv2d_bn(x, 32, 3, 3, padding="valid", dropout_rate=dropout_rate)
        x = self.conv2d_bn(x, 64, 3, 3, dropout_rate=dropout_rate)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding="valid", dropout_rate=dropout_rate)
        x = self.conv2d_bn(x, 192, 3, 3, padding="valid", dropout_rate=dropout_rate)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=dropout_rate)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=dropout_rate)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1, dropout_rate=dropout_rate)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed0",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=dropout_rate)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=dropout_rate)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, dropout_rate=dropout_rate)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed1",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 2: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=dropout_rate)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=dropout_rate)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, dropout_rate=dropout_rate)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed2",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(
            x, 384, 3, 3, strides=(2, 2), padding="valid", dropout_rate=dropout_rate
        )

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=dropout_rate)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl,
            96,
            3,
            3,
            strides=(2, 2),
            padding="valid",
            dropout_rate=dropout_rate,
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed3")(
            [branch3x3, branch3x3dbl, branch_pool]
        )

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1, dropout_rate=dropout_rate)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7, dropout_rate=dropout_rate)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=dropout_rate)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1, dropout_rate=dropout_rate)
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 128, 7, 1, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 128, 1, 7, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 128, 7, 1, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 192, 1, 7, dropout_rate=dropout_rate
        )

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=dropout_rate)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed4",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1, dropout_rate=dropout_rate)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7, dropout_rate=dropout_rate)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=dropout_rate)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1, dropout_rate=dropout_rate)
            branch7x7dbl = self.conv2d_bn(
                branch7x7dbl, 160, 7, 1, dropout_rate=dropout_rate
            )
            branch7x7dbl = self.conv2d_bn(
                branch7x7dbl, 160, 1, 7, dropout_rate=dropout_rate
            )
            branch7x7dbl = self.conv2d_bn(
                branch7x7dbl, 160, 7, 1, dropout_rate=dropout_rate
            )
            branch7x7dbl = self.conv2d_bn(
                branch7x7dbl, 192, 1, 7, dropout_rate=dropout_rate
            )

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(
                branch_pool, 192, 1, 1, dropout_rate=dropout_rate
            )
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(5 + i),
            )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7, dropout_rate=dropout_rate)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=dropout_rate)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 192, 7, 1, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 192, 1, 7, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 192, 7, 1, dropout_rate=dropout_rate
        )
        branch7x7dbl = self.conv2d_bn(
            branch7x7dbl, 192, 1, 7, dropout_rate=dropout_rate
        )

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=dropout_rate)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed7",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)
        branch3x3 = self.conv2d_bn(
            branch3x3,
            320,
            3,
            3,
            strides=(2, 2),
            padding="valid",
            dropout_rate=dropout_rate,
        )

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=dropout_rate)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7, dropout_rate=dropout_rate)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1, dropout_rate=dropout_rate)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3,
            192,
            3,
            3,
            strides=(2, 2),
            padding="valid",
            dropout_rate=dropout_rate,
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed8")(
            [branch3x3, branch7x7x3, branch_pool]
        )

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1, dropout_rate=dropout_rate)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1, dropout_rate=dropout_rate)
            branch3x3_1 = self.conv2d_bn(
                branch3x3, 384, 1, 3, dropout_rate=dropout_rate
            )
            branch3x3_2 = self.conv2d_bn(
                branch3x3, 384, 3, 1, dropout_rate=dropout_rate
            )
            branch3x3 = layers.Concatenate(axis=channel_axis, name="mixed9_" + str(i))(
                [branch3x3_1, branch3x3_2]
            )

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1, dropout_rate=dropout_rate)
            branch3x3dbl = self.conv2d_bn(
                branch3x3dbl, 384, 3, 3, dropout_rate=dropout_rate
            )
            branch3x3dbl_1 = self.conv2d_bn(
                branch3x3dbl, 384, 1, 3, dropout_rate=dropout_rate
            )
            branch3x3dbl_2 = self.conv2d_bn(
                branch3x3dbl, 384, 3, 1, dropout_rate=dropout_rate
            )
            branch3x3dbl = layers.Concatenate(axis=channel_axis)(
                [branch3x3dbl_1, branch3x3dbl_2]
            )

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(
                branch_pool, 192, 1, 1, dropout_rate=dropout_rate
            )
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(9 + i),
            )([branch1x1, branch3x3, branch3x3dbl, branch_pool])

        inputs = img_input
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)

        model = training.Model(inputs, x, name="inception_v3")

        if weight_decay_rate is not None:
            l2_regularizer = regularizers.l2(weight_decay_rate)
            for layer in model.layers:
                if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                    model.add_loss(lambda: l2_regularizer(layer.kernel))
                if hasattr(layer, "bias_regularizer") and layer.use_bias:
                    model.add_loss(lambda: l2_regularizer(layer.bias))

        return model

    def load_imagenet_weights(self):
        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        weights_path = data_utils.get_file(
            "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="bcbd6486424b2319ff4ef7d526e38f63",
        )

        with h5py.File(weights_path, "r") as f:
            # changing model layer names
            for layer in self.model.layers:
                # print(f"$$$$$$$$before: {layer.name}")
                if layer.name in ["input_1", "mixed9_0", "mixed9_1"]:
                    continue
                if layer.name in [
                    "conv2d",
                    "batch_normalization",
                    "activation",
                    "max_pooling2d",
                    "average_pooling2d",
                    "dropout",
                    "concatenate",
                ]:
                    layer._name = layer.name + "_1"
                else:
                    idx = layer.name.rfind("_")
                    if idx != -1:
                        try:
                            layer._name = layer.name[: idx + 1] + str(
                                int(layer.name[idx + 1 :]) + 1
                            )
                            # print(f"$$$$$$$$AFTERR: {layer.name}")
                        except:
                            continue

            # model_layer_name = []
            # for layer in self.model.layers:
            #   model_layer_name.append(layer.name)

            file_layer_names = load_attributes_from_hdf5_group(f, "layer_names")

            # print("########model:")
            # print(sorted(model_layer_name))
            # print("########file:")
            # print(file_layer_names)

            for layer_name in file_layer_names:
                if (layer_name != "average_pooling2d_10") and (
                    "batch_normalization" not in layer_name
                ):
                    # print(layer_name)
                    layer = self.model.get_layer(layer_name)
                    weight_values = load_subset_weights_from_hdf5_group(f[layer_name])
                    layer.set_weights(weight_values)

        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(self.num_classes, activation="sigmoid")(x)

        self.model = training.Model(self.model.input, x)
