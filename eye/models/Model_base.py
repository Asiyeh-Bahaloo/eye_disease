import tensorflow as tf
from abc import abstractmethod


class ModelBase:
    def __init__(self, input_shape, metrics, weights_path):
        """
        Initializer for every model base.

        Parameters
        ----------
        input_shape : tuple
            the shape of you input image
        metrics : list
            metrics you want to track in your model. As the same you pass to keras.fit
        weights_path : str
            Direct Path to model weights file (.h5)
        """
        self.input_shape = input_shape
        self.metrics = metrics
        self.weights_path = weights_path

    def show_summary(self, model):
        """
        Prints the summary of given model.

        Parameters
        ----------
        model : keras.model
            The model you want to print its summary
        """
        model.summary()

    def plot_summary(self, model, file_name):
        """
        Plot a model's architecture.

        Parameters
        ----------
        model : keras.model
            The model you want to plot its architecture.
        file_name : str
            Direct path to the file you want to save the result.
        """
        tf.keras.utils.plot_model(
            model, to_file=file_name, show_shapes=True, show_layer_names=True
        )

    @abstractmethod
    def compile(self):
        pass
