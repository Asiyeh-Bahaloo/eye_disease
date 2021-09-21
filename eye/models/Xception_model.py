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
        self.base_model = xception.Xception

    def load_xception_weight(self, url):
        """load_weight function used for loading pretrained weight

        These types of weights are perfectly tailored to the personalized model

        Parameters
        ----------
         url : str
            it is the address of pretrained weights that we use
        """

        self.base_model.load_weights(url)

    def load_imagenet_weights(self, url):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part

        Parameters
        ----------
        url : str
            it is the address of pretrained weights that we use
        """
        self.base_model = self.base_model(include_top=False)
        self.base_model.load_weights(url)

    def costomizeModel(self, act_func1="relu", num_layer1=1024, act_func2="sigmoid"):
        """costomizeModel this function costomize xception model according to the classify task

        in this function by add some layer with desired node and activation function you can costomize xception model

        Parameters
        ----------
        act_func1 : str, optional
            it indicates the activation function of the first layer after xception model , by default 'relu'
        num_layer1 : int, optional
           the number of nodes in the first layer after xception model, by default 1024
        act_func2 : str, optional
             it indicates the activation function of the second layer after xception model, by default 'sigmoid'

        Returns
        -------
        model object
            it is the customized xception model  but not compiled and trained
        """

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_layer1, activation=act_func1)(x)
        prediction = Dense(self.num_classes, activation=act_func2)(x)
        self.base_model = Model(inputs=self.base_model.input, outputs=prediction)

        return self.base_model
