from keras import backend
from keras.layers import VersionAwareLayers
from .model import KerasClsBaseModel
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import data_utils
from keras.utils import layer_utils

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


class Vgg16(KerasClsBaseModel):
    """Vgg16 a model based on VGG16 model to detect disease

    Parameters
    ----------
    num_classes : int
        number of classes of the classification task
    """

    def __init__(self, num_classes, input_shape):
        """__init__ set number of classes and builds model architecture

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        """
        # super().__init__(num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.build(self.num_classes, self.input_shape)

    def build(self, num_classes, input_shape):
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

        img_input = layers.Input(shape=input_shape)
        # Block 1
        x = layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv1"
        )(img_input)
        x = layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv2"
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv1"
        )(x)
        x = layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv2"
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv1"
        )(x)
        x = layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv2"
        )(x)
        x = layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv3"
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv1"
        )(x)
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv2"
        )(x)
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv3"
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv1"
        )(x)
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv2"
        )(x)
        x = layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv3"
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)
        x = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

        inputs = img_input
        model = training.Model(inputs, x, name="vgg16")

        return model

    def load_imagenet_weights(self):
        """loads imagenet-pretrained weight into model"""

        x = self.model.layers[-5].output
        self.model = training.Model(self.model.input, x)

        weights_path = data_utils.get_file(
            "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="6d6bbae143d832006294945121d1f1fc",
        )
        self.model.load_weights(weights_path)

        x = self.model.layers[-1].output
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)
        x = layers.Dense(self.num_classes, activation="sigmoid", name="predictions")(x)

        self.model = training.Model(self.model.input, x)
