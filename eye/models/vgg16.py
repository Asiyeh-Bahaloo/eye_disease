from tensorflow.keras import models, layers
from .model import KerasClsBaseModel
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras import regularizers


WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)


class Vgg16(KerasClsBaseModel):
    """Vgg16 a model based on VGG16 model to detect disease

    Parameters
    ----------
    num_classes : int
        number of classes of the classification task
    """

    def __init__(
        self, num_classes, input_shape, dropout_rate=None, weight_decay_rate=None
    ):
        """__init__ set number of classes and builds model architecture

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        """
        super().__init__(num_classes, input_shape, dropout_rate, weight_decay_rate)

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

        Returns
        -------
        model [keras sequential model]
              model's Architecture
        """

        model = models.Sequential()
        # Block 1
        layer = layers.Conv2D(
            input_shape=input_shape,
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

        model.add(layer)

        # Block 2
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

        model.add(layer)

        # Block 3
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

        model.add(layer)

        # Block 4
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

        model.add(layer)

        # Block 5
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )

        model.add(layer)
        if dropout_rate is not None:
            layer = layers.Dropout(dropout_rate)
            model.add(layer)

        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

        model.add(layer)

        layer = layers.Flatten()

        model.add(layer)
        layer = layers.Dense(4096, activation="relu")

        model.add(layer)
        layer = layers.Dense(4096, activation="relu")

        model.add(layer)
        layer = layers.Dense(num_classes, activation="sigmoid")
        model.add(layer)

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

        self.model.pop()
        layer = layers.Dense(1000, activation="softmax")
        layer.trainable = False
        self.model.add(layer)

        weights_path = data_utils.get_file(
            "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
            file_hash="64373286793e3c8b2b4e3219cbf3544b",
        )

        self.model.load_weights(weights_path)

        self.model.pop()

        # Add new dense layer
        self.model.add(layers.Dense(self.num_classes, activation="sigmoid"))
