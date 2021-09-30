from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3

from .model import KerasClsBaseModel


class InceptionV3(KerasClsBaseModel):
    def __init__(self, num_classes):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """
        super().__init__(num_classes)

    def build(self, num_classes, pretrained_backbone=None):
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

        if pretrained_backbone is not None:
            backbone = pretrained_backbone
        else:
            backbone = inception_v3.InceptionV3(weights=None, include_top=False)
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=backbone.input, outputs=x)

        return model

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """
        backbone = inception_v3.InceptionV3(weights="imagenet", include_top=False)
        self.model = self.build(self.num_class, backbone)
