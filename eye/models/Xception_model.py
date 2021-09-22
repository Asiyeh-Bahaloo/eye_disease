import tensorflow as tf
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.backend import dtype


class xceptionModel:
    def __init__(self, num_classes):
        """__init__  is  constructor function of xceptionModel class

        this class creat a customized xception model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        base_model : model object, optional
            it is the primary model architecture that we want to  customize it , by default xception.Xception
        """
        self.num_classes = num_classes
        self.base_model = xception.Xception(weights=None, include_top=False)
        temp = self.base_model
        self.base_model = self.base_model.output
        self.base_model = GlobalAveragePooling2D()(self.base_model)
        self.base_model = Dense(1024, activation="relu")(self.base_model)
        prediction = Dense(self.num_classes, activation="sigmoid")(self.base_model)
        self.base_model = Model(inputs=temp.input, outputs=prediction)

    def load_xception_weight(self, url):
        """load_weight function used for loading pretrained weight

        These types of weights are perfectly tailored to the personalized model

        Parameters
        ----------
         url : str
            it is the address of pretrained weights that we use
        """

        self.base_model.load_weights(url)

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part

        Parameters
        ----------
        url : str
            it is the address of pretrained weights that we use
        """
        self.base_model = xception.Xception(weights="imagenet", include_top=False)
        temp = self.base_model
        self.base_model = self.base_model.output
        self.base_model = GlobalAveragePooling2D()(self.base_model)
        self.base_model = Dense(1024, activation="relu")(self.base_model)
        prediction = Dense(self.num_classes, activation="sigmoid")(self.base_model)
        self.base_model = Model(inputs=temp.input, outputs=prediction)
