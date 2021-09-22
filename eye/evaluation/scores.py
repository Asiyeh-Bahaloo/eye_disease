from sklearn import metrics


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
