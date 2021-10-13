from sklearn import metrics
from keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import numpy as np


def kappa_score(gt, pred, threshold=0.5):
    """
    returns kappa score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        calculated kappa score
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.cohen_kappa_score(gt_flat, pred_flat > threshold)


def f1_score(gt, pred, threshold=0.5):
    """
    returns f1 score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        calculated f1 score
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.f1_score(gt_flat, pred_flat > threshold, average="micro")


def auc_score(gt, pred):
    """
    returns AUC score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)

    Returns
    -------
    float
        calculated AUC score
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    try:
        return metrics.roc_auc_score(gt_flat, pred_flat)
    except ValueError:
        return 0.0


def final_score(gt, pred, threshold=0.5):
    """
    This function return mean of the kappa, f1, AUC scores.


    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        final score: mean of the kappa, f1, AUC scores.
    """

    kappa = kappa_score(gt, pred, threshold)
    f1 = f1_score(gt, pred, threshold)
    auc = auc_score(gt, pred)
    return (kappa + f1 + auc) / 3.0


def specificity(y_true, y_pred):
    """specificity function calculate  the specificity of model

    this function with getting true label and predicted label of model calculates  the specificity
    also we can pass this function as parameters to compile function

    Parameters
    ----------
    y_true : numpy array
        the matrix of true label
    y_pred : numpy array
        the matrix of predicted label

    Returns
    -------
    tf.tensor
        it is the only number of specificity
    """
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fp = tf.cast(fp, tf.float32)
    tn = tf.cast(tn, tf.float32)
    return tn / (tn + fp + K.epsilon())


def sensitivity(y_true, y_pred):
    """sensitivity function calculate  the specificity of sensitivity

    this function with getting true label and predicted label of model calculates  the sensitivity
    also we can pass this function as parameters to compile function

    Parameters
    ----------
    y_true : numpy array
        the matrix of true label
    y_pred : numpy array
        the matrix of predicted label

    Returns
    -------
    tf.tensor
        it is the only number of sensitivity
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = tf.cast(possible_positives, tf.float32)
    return true_positives / (possible_positives + K.epsilon())


def get_specific_metrics(x_test, y_test, model, metrics, loss):
    """get_specific_metrics get desired metrics for given model

    Parameters
    ----------
    x_test : tensor
        testing input as a tensor
    y_test : tensor
        actual labels as a tensor
    model : keras.Model
        the model that will be evaluated
    metrics : list, optional
        list of desired metrics
    loss : keras.loss
        loss function to evaluate model

    Returns
    -------
    list
        loss and desired metrics in a list respectively
    """

    # getting metrics that will be calculated by model.evaluate
    score, metric_names = (
        model.evaluate(X=x_test, Y=y_test, metrics=metrics, loss=loss),
        model.model.metrics_names,
    )

    return score, metric_names


def accuracy_score(gt, pred, threshold=0.5):
    """
    returns accuracy score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        calculated accuracy
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.accuracy_score(gt_flat, pred_flat > threshold)


def precision_score(gt, pred, threshold=0.5):
    """
    returns precision score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        calculated precision score
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.precision_score(gt_flat, pred_flat > threshold)


def recall_score(gt, pred, threshold=0.5):
    """
    returns recall score based on the ground trouth and predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        ground trouth vector having shape (m,8)
    pred : numpy.ndarray
        prediction vector having shape (m,8)
    threshold : float, optional
        threshold used to evaluate prediction outputs, by default 0.5

    Returns
    -------
    float
        calculated recall score
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.recall_score(gt_flat, pred_flat > threshold)


def accuracy_per_class(label):
    """A wrapper function that calculates the accuracy of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the accuracy of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def accuracy_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return accuracy_score(y_true, y_pred)

    accuracy_per_label.__name__ = f"accuracyForLabel{class_names[label]}"
    return accuracy_per_label


def precision_per_class(label):
    """A wrapper function that calculates the precision score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the precision score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def precision_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return precision_score(y_true, y_pred)

    precision_per_label.__name__ = f"precisionForLabel{class_names[label]}"
    return precision_per_label


def recall_per_class(label):
    """A wrapper function that calculates the recall score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the recall score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def recall_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return recall_score(y_true, y_pred)

    recall_per_label.__name__ = f"recallForLabel{class_names[label]}"
    return recall_per_label


def kappa_per_class(label):
    """A wrapper function that calculates the kappa score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the kappa score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def kappa_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return kappa_score(y_true, y_pred)

    kappa_per_label.__name__ = f"kappaForLabel{class_names[label]}"
    return kappa_per_label


def f1_per_class(label):
    """A wrapper function that calculates the f1 score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the f1 score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def f1_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return f1_score(y_true, y_pred)

    f1_per_label.__name__ = f"f1ForLabel{class_names[label]}"
    return f1_per_label


def auc_per_class(label):
    """A wrapper function that calculates the auc score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the auc score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def auc_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return auc_score(y_true, y_pred)

    auc_per_label.__name__ = f"aucForLabel{class_names[label]}"
    return auc_per_label


def final_per_class(label):
    """A wrapper function that calculates the final score of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the final score of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def final_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return final_score(y_true, y_pred)

    final_per_label.__name__ = f"finalForLabel{class_names[label]}"
    return final_per_label


def specificity_per_class(label):
    """A wrapper function that calculates the specificity of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the specificity of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def specificity_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return specificity(y_true, y_pred)

    specificity_per_label.__name__ = f"specificityForLabel{class_names[label]}"
    return specificity_per_label


def sensitivity_per_class(label):
    """A wrapper function that calculates the sensitivity of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the sensitivity of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def sensitivity_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return sensitivity(y_true, y_pred)

    sensitivity_per_label.__name__ = f"sensitivityForLabel{class_names[label]}"
    return sensitivity_per_label


def loss_per_class(label):
    """A wrapper function that calculates the loss of each disease based on the label.

    Parameters
    ----------
    label : int
        the class number of the disease

    returns
    -------
    function
        function that calculates the loss of each disease based on the label.
    """
    class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def loss_per_label(y_true, y_pred):
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return K.binary_crossentropy(y_true, y_pred)

    loss_per_label.__name__ = f"lossForLabel{class_names[label]}"
    return loss_per_label
