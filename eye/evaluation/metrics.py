from sklearn import metrics
from keras import backend as K
import tensorflow as tf


def kappa_score(gt, pred, threshold=0.5):
    """
    returns kappa score based on the grounf trouth and predictions.

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
    returns f1 score based on the grounf trouth and predictions.

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
    returns AUC score based on the grounf trouth and predictions.

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
    return metrics.roc_auc_score(gt_flat, pred_flat)


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
