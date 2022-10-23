import tensorflow as tf
import streamlit as st


from eye.utils.utils import load_data
from eye.UI.ui_function import (
    set_model_architecture,
    set_weight,
    set_optimizer,
)


def app(data_path):
    """app "app this function used for running home page of ui

     in the main app when you want to diaplay home page you must call this function

    Parameters
    ----------
    data_path : str
        path of dataset

    Returns
    -------
    model object
        the model object after training
        if you do not biuld and train the model the function return None
    history object
        the history object of training process
        if you do not biuld and train the model the function return None
    """

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(data_path)

    header = st.container()
    dataset = st.container()
    models = st.container()
    train = st.container()

    with header:
        st.title("WELCOME TO EYE DISEASE PREDICTOR!")
        st.write(
            "there you can choose the arbitrary model and train it for prediction of eye diseases of fundus image"
        )

    st.title("WELCOME TO EYE DISEASE PREDICTOR!")
    st.write(
        "there you can choose the arbitrary model and train it for prediction of eye diseases of fundus image"
    )

    # in this container we show data and according info about them
    with dataset:

        st.header("FUNDUS IMAGE")

        # display data samples in firste of the page
        temp = x_train[0:3]
        temp2 = x_train[3:6]
        st.image(temp)
        st.image(temp2)

    # in this container we display some UI for setting model and the other params about optimizer,score,...
    # and the other thing about biulding and trsainig model

    with models:
        # here we set ui for creating model
        model_obj, model_architecture = set_model_architecture(
            num_classes=8, input_shape=x_train[0].shape
        )

        # here we set ui for setting  primary weight
        if model_obj is not None:
            set_weight(model_obj=model_obj, model_architecture=model_architecture)

        # here we set ui for setting optimizer
        sgd = set_optimizer(model_obj, optimizer_type="sgd")

        # setting loss finction
        loss_function = st.selectbox(
            "what kind of loss finction do you want to use?",
            ("binary_crossentropy", "defualt"),
        )

        # setting batch_size,patience,epochs
        batch_size = st.number_input("Insert a batch_size", value=2)
        patience = st.number_input("Insert a patience", value=2)
        epochs = st.number_input("Insert a epochs", value=1)

        # define the metrics for passing to train function
        defined_metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

        # define the callback for passing to train function
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=1
        )

    with train:
        st.header("TRAIN IT")
        if st.button("TRAIN"):
            his = model_obj.train(
                epochs=epochs,
                loss=loss_function,
                metrics=defined_metrics,
                callbacks=[callback],
                optimizer=sgd,
                X=x_train,
                Y=y_train,
                X_val=x_val,
                Y_val=y_val,
                batch_size=batch_size,
            )
            return his, model_obj
        else:
            return None, None
