import tensorflow as tf
from .preprocess import resize_image, remove_padding, ben_graham
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as preprocess_input_vgg16,
)
from tensorflow.keras.applications.vgg19 import (
    preprocess_input as preprocess_input_vgg19,
)
from tensorflow.keras.applications.xception import (
    preprocess_input as preprocess_input_xception,
)
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as preprocess_input_inception_v3,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as preprocess_input_inception_resnet_v2,
)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Resize(object):
    def __init__(self, image_shape, keep_aspect_ration):
        self.image_shape = image_shape
        self.keep_aspect_ration = keep_aspect_ration

    def __call__(self, image):
        img = resize_image(
            img=image,
            image_shape=self.image_shape,
            keep_aspect_ration=False,
        )
        return img


class BenGraham(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        img = ben_graham(image, self.scale)
        return img


class RemovePadding(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img = remove_padding(img=image)
        return img


class RandomShift(object):
    def __init__(self, wrg, hrg):
        """

        Parameters
        ----------
        wrg : float
            Width shift range, as a float fraction of the width.
        hrg : float
            Height shift range, as a float fraction of the height.
        """
        self.wrg = wrg
        self.hrg = hrg

    def __call__(self, image):
        img = tf.keras.preprocessing.image.random_shift(
            x=image, wrg=self.wrg, hrg=self.hrg
        )
        return img


class RandomFlipLR(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img = tf.image.random_flip_left_right(image)
        return img


class RandomFlipUD(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img = tf.image.random_flip_up_down(image)
        return img


class KerasPreprocess(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, image):
        if self.model_name == "vgg16":
            img = preprocess_input_vgg16(image)
        elif self.model_name == "vgg19":
            img = preprocess_input_vgg19(image)
        elif self.model_name == "xception":
            img = preprocess_input_xception(image)
        elif self.model_name == "inception":
            img = preprocess_input_inception_v3(image)
        elif self.model_name == "resnet":
            img = preprocess_input_inception_resnet_v2(image)

        return img
