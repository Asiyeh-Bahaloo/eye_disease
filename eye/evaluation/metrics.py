from numpy import mean
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf


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
    if not isinstance(gt, np.ndarray):
        gt = gt.numpy()
        pred = pred.numpy()

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.cohen_kappa_score(gt_flat, pred_flat > threshold)


def kappa_per_class(label, threshold=0.5):
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
        return kappa_score(y_true, y_pred, threshold)

    kappa_per_label.__name__ = f"kappaForLabel{class_names[label]}"
    return kappa_per_label


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


def f1_per_class(label, threshold=0.5):
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
        return f1_score(y_true, y_pred, threshold)

    f1_per_label.__name__ = f"f1ForLabel{class_names[label]}"
    return f1_per_label


def micro_f1_score(y_true, y_pred, threshold=0.5):
    """
        calculates f1 for each label and returns mean of them
    Parameters
    ----------
    y_true : list of floats or ndarray
        true labels
    y_pred : list of floats or ndarray
        predicted labels
    Returns
    -------
    f1:float
        f1_score with average of 'micro'
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    y_pred = (y_pred > threshold).astype("int")
    return metrics.f1_score(y_true, y_pred, average="micro")


def auc_score(gt, pred, threshold=0.5):
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
    pred_flat = pred_flat > threshold
    try:
        return metrics.roc_auc_score(gt_flat, pred_flat)
    except ValueError:
        return 0.0


def auc_per_class(label, threshold=0.5):
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
        return auc_score(y_true, y_pred, threshold)

    auc_per_label.__name__ = f"aucForLabel{class_names[label]}"
    return auc_per_label


def micro_auc(y_true, y_pred, threshold=0.5):
    """
    calculates auc
    Parameters
    ----------
    y_true:list of floats or ndarray
        true labels
    y_pred: list of floats or ndarray
        predicted labels
    Returns
    -------
    auc:float
        auc with average of "micro"
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    y_pred = (y_pred > threshold).astype("int")

    return metrics.roc_auc_score(y_true, y_pred, average="micro")


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
    if not isinstance(gt, np.ndarray):
        gt = gt.numpy()
        pred = pred.numpy()

    kappa = kappa_score(gt, pred, threshold)
    f1 = f1_score(gt, pred, threshold)
    auc = auc_score(gt, pred, threshold)

    return (kappa + f1 + auc) / 3.0


def final_per_class(label, threshold=0.5):
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
        return final_score(y_true, y_pred, threshold)

    final_per_label.__name__ = f"finalForLabel{class_names[label]}"
    return final_per_label


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
    if not isinstance(gt, np.ndarray):
        gt = gt.numpy()
        pred = pred.numpy()

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    return metrics.accuracy_score(gt_flat, pred_flat > threshold)


def accuracy_per_class(label, threshold=0.5):
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
        return accuracy_score(y_true, y_pred, threshold)

    accuracy_per_label.__name__ = f"accuracyForLabel{class_names[label]}"
    return accuracy_per_label


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
    return metrics.precision_score(gt_flat, pred_flat > threshold, zero_division=0)


def precision_per_class(label, threshold=0.5):
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
        return precision_score(y_true, y_pred, threshold)

    precision_per_label.__name__ = f"precisionForLabel{class_names[label]}"
    return precision_per_label


def micro_precision(y_true, y_pred, threshold=0.5):
    """
        calculates precision of each labels and returns mean of them
    Parameters
    ----------
    y_true : list of floats or ndarray
        true labels
    y_pred : lost of floats or ndarray
        predicted labels
    Returns
    -------
    recall: float
        precision with average of 'micro'
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    y_pred = (y_pred > threshold).astype("int")

    return metrics.precision_score(y_true, y_pred, average="micro", zero_division=0)


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
    return metrics.recall_score(gt_flat, pred_flat > threshold, zero_division=0)


def recall_per_class(label, threshold=0.5):
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
        return recall_score(y_true, y_pred, threshold)

    recall_per_label.__name__ = f"recallForLabel{class_names[label]}"
    return recall_per_label


def micro_recall(y_true, y_pred, threshold=0.5):
    """
        calculates recall of each labels and returns mean of them
    Parameters
    ----------
    y_true : list of floats or ndarray
        true labels
    y_pred : lost of floats or ndarray
        predicted labels
    Returns
    -------
    recall: float
        recall with average of 'micro'
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    y_pred = (y_pred > threshold).astype("int")
    return metrics.recall_score(y_true, y_pred, average="micro", zero_division=0)


def specificity_per_class(label, threshold=0.5):
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
        return specificity(y_true, y_pred, threshold)

    specificity_per_label.__name__ = f"specificityForLabel{class_names[label]}"
    return specificity_per_label


def specificity(y_true, y_pred, threshold=0.5):
    """
    calculate specificity
    Parameters
    ----------
    y_true: list of floats of numpy.ndarray
        true labels
    y_pred: list of floats of numpy.ndarray
        predicted labels
    Returns
    -------
    macro_specificity: float
       specificities
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = (y_pred > threshold).astype("int")

    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) != 0 else 0

    return specificity


def micro_specificity(y_true, y_pred, threshold=0.5):
    """

    Parameters
    ----------
    y_true : np.ndarray / list of numbers between [0, 1]
         actual labels
    y_pred : np.ndarray / list of numbers between [0, 1]
        predicted labels
    Returns
    -------
    float: specificity with micro average
    """
    if not isinstance(y_pred, np.ndarray):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    y_true = y_true.flatten()
    y_pred = (y_pred > threshold).astype("int")
    y_pred = y_pred.flatten()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, _, _ = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) != 0 else 0

    return specificity


def sensitivity_per_class(label, threshold=0.5):
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
        return micro_sensitivity(y_true, y_pred, threshold)

    sensitivity_per_label.__name__ = f"sensitivityForLabel{class_names[label]}"
    return sensitivity_per_label


def micro_sensitivity(y_true, y_pred, threshold=0.5):
    """

    Parameters
    ----------
    y_true : np.ndarray / list of numbers between [0, 1]
         actual labels
    y_pred : np.ndarray / list of numbers between [0, 1]
        predicted labels
    Returns
    -------
    float: sensitivity with micro average
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = tf.cast(possible_positives, tf.float32)
    true_positives = tf.cast(true_positives, tf.float32)
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
        bce = BinaryCrossentropy(from_logits=True)
        y_pred = y_pred.numpy()[:, label]
        y_true = y_true.numpy()[:, label]
        return bce(y_true, y_pred).numpy()

    loss_per_label.__name__ = f"lossForLabel{class_names[label]}"
    return loss_per_label


def loss(
    y_true,
    y_pred,
):
    """A function that calculates the loss of Model.

        Parameters
    ----------
    y_true: list of floats of numpy.ndarray
        true labels
    y_pred: list of floats of numpy.ndarray
        predicted labels
    Returns
    -------
        loss
    """
    bce = BinaryCrossentropy(from_logits=True)
    tensor_y_train = tf.convert_to_tensor(y_true, np.float32)
    tensor_pred_train_y = tf.convert_to_tensor(y_pred, np.float32)
    return bce(tensor_y_train, tensor_pred_train_y).numpy()
