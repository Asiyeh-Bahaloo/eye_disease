from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD


class Vgg19:
    """Implements the VGG19 architecture.

    Attributes
    ----------
    input_shape : tuple
        The shape of the input images(e.g. (224, 224, 3)).
    metrics : list
        Metrics to evaluate the model based on them.
    model : models
        The model architecture.

    Methods
    -------
    get_model()
        Builds the architecture of model.
    compile(loss, lr)
        Compiles the model.
    load_VGG19_weights(weights_path)
        Loads prepared VGG19 weights into the raw model.
    load_ImageNet_weights(weights_path)
        Loads prepared ImageNet weights into the raw model.
    """

    def __init__(self, input_shape, metrics):
        """Constructs the VGG19 model.
        Parameters
        ----------
        input_shape : tuple
            The shape of the input images(e.g. (224, 224, 3)).
        metrics : list
            Metrics to evaluate the model based on them.
        """
        self.input_shape = input_shape
        self.metrics = metrics
        self.model = None
        self.get_model()

    def get_model(self):
        """Builds the model architecture."""
        model = models.Sequential()
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
        model.add(layer)
        layer = layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 2
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 3
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 4
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 5
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        layer = layers.Flatten()
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = trainable
        model.add(layer)
        # layer = layers.Dropout(0.5)
        # layer.trainable = True
        # model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = trainable
        model.add(layer)
        # layer = layers.Dropout(0.5)
        # layer.trainable = True
        # model.add(layer)
        model.add(layers.Dense(8, activation="sigmoid"))
        self.model = model
        # return model

    def compile(self, loss, lr):
        """Compiles the model.
        Parameters
        -------
        loss : str
            Loss function in compiling the model.
        lr : float
            The learing rate of optimizer.
        """
        sgd = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=False)
        print("Configuration Start -------------------------")
        print(sgd.get_config())
        print("Configuration End -------------------------")
        self.model.compile(optimizer=sgd, loss=loss, metrics=self.metrics)

    def load_VGG19_weights(self, weights_path):
        """Loads the VGG19 weights into the model.

        Parameters
        ----------
        weights_path : str
            Path of weights to be loaded into the model.
        """
        self.model.load_weights(weights_path)

    def load_imagenet_weights(self, weights_path):
        """Loads the ImageNet weights into the model.

        Parameters
        ----------
        weights_path : str
            Path of weights to be loaded into the model.
        """
        self.model.pop()
        layer = layers.Dense(1000, activation="softmax")
        self.model.add(layer)
        # Transfer learning, load previous weights
        self.model.load_weights(weights_path)
        # Remove last layer
        self.model.pop()
        self.model.add(layers.Dense(8, activation="sigmoid"))
