
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.applications.inception_v3 import InceptionV3


class Inception_V3:
    def __init__(self, num_classes):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        base_model : model object, optional
            it is the primary model architecture that we want to  customize it , by default inception.Inception
        """
        self.num_classes = num_classes
        self.base_model = InceptionV3(weights=None, include_top=False)
        temp = self.base_model
        self.base_model = self.base_model.output
        self.base_model = GlobalAveragePooling2D()(self.base_model)
        self.base_model = Dense(1024, activation="relu")(self.base_model)
        prediction = Dense(self.num_classes, activation="sigmoid")(self.base_model)
        self.base_model = Model(inputs=temp.input, outputs=prediction)

    def load_inception_weight(self, path):
        """load_weight function used for loading pretrained weight

        These types of weights are perfectly tailored to the personalized model

        Parameters
        ----------
         path : str
            it is the address of pretrained weights that we use
        """

        self.base_model.load_weights(path)


    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """
        self.base_model = InceptionV3(weights="imagenet", include_top=False)
        temp = self.base_model
        self.base_model = self.base_model.output
        self.base_model = GlobalAveragePooling2D()(self.base_model)
        self.base_model = Dense(1024, activation="relu")(self.base_model)
        prediction = Dense(self.num_classes, activation="sigmoid")(self.base_model)
        self.base_model = Model(inputs=temp.input, outputs=prediction)
