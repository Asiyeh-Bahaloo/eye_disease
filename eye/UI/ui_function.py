import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import streamlit as st


from eye.models.vgg16 import Vgg16
from eye.models.vgg19 import Vgg19
from eye.models.xception import Xception
from eye.models.resnet_v2 import ResnetV2
from eye.models.inception_v3 import InceptionV3


def set_model_architecture(num_classes=8, input_shape=None):
    """set_model_architecture function biuld model according to the architecture that the user chosses

    this function set the object from models class

    Parameters
    ----------
    num_classes : int, optional
        the number of output of model(number classifying labels), by default 8
    input_shape : Tuple, optional
        this Tuple indicates the size of input image of model, by default None

    Returns
    -------
    model object
        the object of model class
    str
        string that indicate the model architecture  name
    """
    # introdiction to this part
    st.header("model architecture & hyperparameter")
    st.text(
        "   here we can choose type model architecture and some  hyperparameters of model "
    )

    # a selsct box for selecting  model_architecture
    model_architecture = st.selectbox(
        "Which architecture do you want to use?",
        ("None", "vgg16", "vgg19", "Resnet_v2", "Inception_v3", "xeption"),
    )

    # here we check the selected model and  creat a object of the class of  selected architecture
    if model_architecture == "None":
        model_obj = None
    elif model_architecture == "vgg16":
        model_obj = None
        model_obj = Vgg16(num_classes=num_classes, input_shape=input_shape)
    elif model_architecture == "vgg19":
        model_obj = None
        model_obj = Vgg19(num_classes=num_classes, input_shape=input_shape)
    elif model_architecture == "xeption":
        model_obj = None
        model_obj = Xception(num_classes=num_classes)
    elif model_architecture == "Resnet_v2":
        model_obj = None
        model_obj = ResnetV2(num_classes=num_classes)
    elif model_architecture == "Inception_v3":
        model = None
        model_obj = InceptionV3(num_classes=num_classes)
    else:
        st.write("pleas choose the model architecture")

    # at last the function return object of model
    return model_obj, model_architecture


def set_weight(model_obj, model_architecture):
    """set_weight function set type of weight

    this function with getting

    Parameters
    ----------
    model_obj : model object
        the object
    model_architecture : str
        the string indicats the type of model architecture
    """

    # a selsct box for selecting  type of weight:random,imagenet,customized weight
    primary_weight = st.selectbox(
        "DO want to use any pre traied primary wight",
        ("random", "imagenet", "customized weight"),
    )

    # setting weight type
    if primary_weight == "customized weight":
        weight_path = st.text_input("wirte the customized weight url")
        if weight_path != "":
            model_obj.load_weights(weight_path)

    elif primary_weight == "imagenet":
        # because for different model ,there are 2 ways for loading imagenet weight :
        # here if the model architecture is vgg16 and vgg19 we set input text fot setting dir of  weight
        if model_architecture == "vgg16" or model_architecture == "vgg19":
            weight_path = st.text_input("wirte the imagenet wight url")
            if weight_path != "":
                model_obj.load_imagenet_weights(weight_path)
        else:
            model_obj.load_imagenet_weights()


def set_optimizer(model_obj, optimizer_type="sgd"):
    """set_optimizer function set optimizer of model object

    [extended_summary]

    Parameters
    ----------
    model_obj : object of model class
         object of model we want to setting optimizer of training
    optimizer_type : str, optional
        type of optimizer, by default "sgd"

    Returns
    -------
    optimizer object
        the optimizer object we should pass it to train finction
    """

    # setting  parameters of sgd optimizer
    lr_rate = st.number_input("enter laerning rate", value=0.001)
    st.write(lr_rate)

    decay_rate = st.number_input("enter decay rate", value=1e-6)
    st.write(decay_rate)

    momentum_rate = st.slider("tuning momentum rate", 0.0, 1.0, value=0.9, step=0.001)
    st.write(momentum_rate)

    nesterov_flag = st.checkbox("set nesterov flag")
    st.write(nesterov_flag)

    sgd = SGD(
        lr=lr_rate,
        decay=decay_rate,
        momentum=momentum_rate,
        nesterov=nesterov_flag,
    )

    return sgd
