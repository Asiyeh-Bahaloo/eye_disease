import tensorflow as tf
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from .model import KerasClsBaseModel


class Xception(KerasClsBaseModel):
    """Xception a model based on Xception model to detect disease

    Parameters
    ----------
    num_classes : int
        number of classes of the classification task
    """

    def __init__(self, num_classes):
        """__init__ set number of classes and builds model architecture

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
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
            backbone = xception.Xception(weights=None, include_top=False)
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=backbone.input, outputs=x)

        return model

    def load_imagenet_weights(self):
        """loads imagenet-pretrained weight into model"""

        backbone = xception.Xception(weights="imagenet", include_top=False)
        self.model = self.build(self.num_class, backbone)
