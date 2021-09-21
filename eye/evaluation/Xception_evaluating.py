from functools import partial
import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow import keras


def predict_xception_model(x_test, y_test, model):
    """predict_xception_model function used for predicting the labels of input image

    this function get model and input data , true output data for indicating

    Parameters
    ----------
    x_test : numpy array
        it is array of the input data(array of RGB image)
    y_test : numpy array
        it is array of output data (a onehot numpy array with size:(#test data x #num_classes))
    model : model object
        a object model that we use it for evaluating test dataset

    Returns
    -------
    list
        a python list that contains the label of each input image,and the label chosses from class_names list that defines blow
    """

    class_names = [
        "Normal",
        "Diabetes",
        "Glaucoma",
        "Cataract",
        "AMD",
        "Hypertension",
        "Myopia",
        "Others",
    ]

    test_predictions_baseline = model.predict(x_test)
    pred = np.argmax(test_predictions_baseline, axis=-1)

    y = np.argmax(y_test, axis=-1)

    pred_label = []
    true_label = []

    for i in range(y.shape[0]):
        pred_label.append(class_names[pred[i]])
        true_label.append(class_names[y[i]])

    return true_label, pred_label
