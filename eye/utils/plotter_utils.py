import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
import seaborn as sns
import matplotlib as mpl
import streamlit as st

from eye.evaluation.metrics import get_specific_metrics


def plot_metrics(history, path):
    """
    This functions plots your defined metrics based on the history you give in the inputs.

    Parameters
    ----------
    history : History.history
        The history of the training process.
    path : str
        Path to the FILE you want to save your result.
    """
    metrics2 = ["loss", "auc", "precision", "recall"]
    for n, metric in enumerate(metrics2):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color="green", label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color="green",
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

    plt.subplots_adjust(
        top=0.97, bottom=0.09, left=0.10, right=0.96, hspace=0.25, wspace=0.26
    )
    plt.savefig(path)
    plt.show()
    plt.close()


def plot_accuracy(history, path):
    """
    This function plots your accuracy changes based on the history you give in the inputs.

    Parameters
    ----------
    history : History.history
        The history of the training process.
    path : str
        Path to the FILE you want to save your result.
    """

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.show()


def plot_input_images(x_train, y_train, class_names):
    """
    This function plots first 30 images of the data given in the input with their labels.

    Parameters
    ----------
    x_train : numpy.ndarray
        Input images to plot. (having shape: (m, x, x, 3) )
    y_train : numpy.ndarray
        Labels of the given images. (having shape: (m, #num_classes) )
    class_names : list
        List of strings that shows the our classes's name.
    """
    plt.figure(figsize=(20, 20))
    for i in range(30):
        plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        classes = ""
        for j in range(8):
            if y_train[i][j] >= 0.5:
                classes = classes + class_names[j] + "\n"
        plt.xlabel(classes, fontsize=14, color="black", labelpad=1)

    plt.subplots_adjust(
        bottom=0.04, right=0.95, top=0.94, left=0.06, wspace=0.56, hspace=0.17
    )
    plt.show()


def calculate_3_largest(arr, arr_size):
    """
    Gives first three largest values and corresponding indices of given 1D numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input 1D numpy array
    arr_size : int
        length of the given array

    Returns
    -------
    float, int
        returns 6 numbers, first three numbers are the largest elements.
        seconde three numbers are the indices of these largest elements in the array.
    """
    assert arr_size > 3, "Invalid Input"
    sorted_array = np.sort(arr)
    sorted_idx = np.argsort(arr)

    return (
        sorted_array[-1],
        sorted_array[-2],
        sorted_array[-3],
        sorted_idx[-1],
        sorted_idx[-2],
        sorted_idx[-3],
    )


def plot_image(pred, label, image, class_names):
    """
    This function plots a single image with the true label and the predictions we have for this image.

    Parameters
    ----------
    pred : numpy.ndarray
        prediction array for this image(it should have #num_class elements).
    label : numpy.ndarray
        ground trouth array for this image(it should have #num_class elements).
    image : numpy.ndarray
        Input Image ( having shape=(x, x, 3) )
    class_names : list
        List of strings that shows the our classes's name.
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)
    label_check = [0, 0, 0, 0, 0, 0, 0, 0]
    ground = ""
    count_true = 0
    predicted_true = 0

    for index in range(8):
        if label[index] >= 0.5:
            count_true = count_true + 1
            ground = ground + class_names[index] + "\n"
            label_check[index] = 1
        if pred[index] >= 0.5:
            predicted_true = predicted_true + 1
            label_check[index] = label_check[index] + 1

    all_match = True
    for index in range(8):
        if label_check[index] == 1:
            all_match = False

    if count_true == predicted_true and all_match:
        color = "green"
    else:
        color = "red"

    first, second, third, i, j, k = calculate_3_largest(pred, 8)
    prediction = "{} {:2.0f}% \n".format(class_names[i], 100 * first)
    if second >= 0.5:
        prediction = prediction + "{} {:2.0f}% \n".format(class_names[j], 100 * second)
    if third >= 0.5:
        prediction = prediction + "{} {:2.0f}% \n".format(class_names[k], 100 * third)
    plt.xlabel("Predicted: {} Ground Truth: {}".format(prediction, ground), color=color)


def plot_value_array(pred, label):
    """
    This function plots a bar chart shows the membership rate of the given image in each class.
    red bar shows our prediction rates.
    green bar shows gound trouth rates.

    Parameters
    ----------
    pred : numpy.ndarray
        prediction array for this image(it should have #num_class elements).
    label : numpy.ndarray
        ground trouth array for this image(it should have #num_class elements).
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    bar_plot = plt.bar(range(8), pred, color="#777777")
    plt.xticks(range(8), ("N", "D", "G", "C", "A", "H", "M", "O"))
    plt.ylim([0, 1])

    for j in range(8):
        if label[j] >= 0.5:
            bar_plot[j].set_color("green")

    for j in range(8):
        if pred[j] >= 0.5 and label[j] < 0.5:
            bar_plot[j].set_color("red")

    def bar_label(rects):
        for rect in rects:
            height = rect.get_height()
            value = height * 100
            if value > 1:
                plt.annotate(
                    "{:2.0f}%".format(value),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    bar_label(bar_plot)


def ensure_test_prediction_exists(predictions):
    """
    This function ensures if the prediction vector has at least a probability more that 0.5

    Parameters
    ----------
    predictions : numpy.ndarray
        prediction array to test(it should have #num_class elements).

    Returns
    -------
    bool
        zero if not valid | one if valid.
    """
    exists = False
    for j in range(8):
        if predictions[j] >= 0.5:
            exists = True
    return exists


def plot_output(pred, y_test, x_test, path_result, class_names):
    """
    This function plots first 10 images of the given data with their labels using bar chart.

    Parameters
    ----------
    pred : numpy.ndarray
        prediction array for this image (having shape: (m, #num_classes) ).
    y_test : numpy.ndarray
        ground trouth array for this image(having shape: (m, #num_classes) ).
    x_test : numpy.ndarray
        The test images. ( having shape: (m, x, x, 3) )
    path_result : str
        Direct path to the file you want to save the results in.
    class_names : list
        List of strings that shows the our classes's name.
    """
    mpl.rcParams["font.size"] = 8
    num_rows = 3
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    j = 0
    i = 0
    while j < num_images:
        if ensure_test_prediction_exists(pred[i]):
            plt.subplot(num_rows, 2 * num_cols, 2 * j + 1)
            plot_image(pred[i], y_test[i], x_test[i], class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * j + 2)
            plot_value_array(pred[i], y_test[i])
            j = j + 1
        i = i + 1
        if i > 10:
            break

    plt.subplots_adjust(
        bottom=0.08, right=0.95, top=0.94, left=0.05, wspace=0.11, hspace=0.56
    )
    plt.savefig(path_result)
    plt.show()


def plot_output_single(pred, y_test, x_test, class_names):
    """
    This function plots the given image with its labels using bar chart.

    Parameters
    ----------
    pred : numpy.ndarray
        prediction array for image (having shape: (#num_classes,) ).
    y_test : numpy.ndarray
        ground trouth array for image(having shape: (#num_classes,) ).
    x_test : numpy.ndarray
        The test images. ( having shape: (x, x, 3) )
    class_names : list
        List of strings that shows the our classes's name.
    """
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(pred, y_test, x_test, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(pred, y_test)
    plt.show()


def plot_confusion_matrix(y_test, pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    y_test : numpy.ndarray
        ground trouth array for this image(having shape: (m, #num_classes) ).
    pred : numpy.ndarray
        prediction array for this image (having shape: (m, #num_classes) ).
    normalize : bool, optional
        Defines if you want to normalize the outputs or not, by default False
    title : str, optional
        Title of the output chart, by default None
    cmap : plt.cm object, optional
        cmap used in atplotlib functions, by default plt.cm.Blues
    """

    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        # xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_ylim(8.0, -1.0)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()


def print_both_confusion_matrix(y_test, pred):
    """
    This function plots both the normalized and non-normalized confusion matrixes.

    Parameters
    ----------
    y_test : numpy.ndarray
        ground trouth array for this image(having shape: (m, #num_classes) ).
    pred : numpy.ndarray
        prediction array for this image (having shape: (m, #num_classes) ).
    """

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(
        y_test,
        pred,
        title="Confusion matrix, without normalization",
    )

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        y_test,
        pred,
        normalize=True,
        title="Normalized confusion matrix",
    )

    plt.show()


def plot_confusion_matrix_sns(y_test, pred, path_result):
    """
    This Function plots the confusion matrix using seaborn library.

    Parameters
    ----------
    y_test : numpy.ndarray
        ground trouth array for this image(having shape: (m, #num_classes) ).
    pred : numpy.ndarray
        prediction array for this image (having shape: (m, #num_classes) ).
    path_result : str
        Direct path to the file you want to save the results in.
    """

    cm = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d")
    ax.set_ylim(8.0, -1.0)
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.savefig(path_result)
    plt.subplots_adjust(
        top=0.94, bottom=0.11, left=0.12, right=1.00, hspace=0.20, wspace=0.18
    )
    plt.show()
    plt.close()


def plot_image_in_UI(pred, image, class_names):
    """plot_image_in_UI function display each image with 3 most possible labels

    this functyin get list of image and say what is the 3 most possible labels greater than 0.5

    Parameters
    ----------
    pred : numpy array
        the array of probability of each label
    image : python list
        list of input image
    class_names : python list
        list of name of each class
    """

    for t in range(len(image)):

        # indicate the 3 most possible label
        first, second, third, i, j, k = calculate_3_largest(np.transpose(pred[t]), 8)
        # indicate which label is greater than 0.5 and filter them
        prediction = "{} {:2.0f}% \n".format(class_names[i], 100 * first)
        if second >= 0.5:
            prediction = prediction + "{} {:2.0f}% \n".format(
                class_names[j], 100 * second
            )
        if third >= 0.5:
            prediction = prediction + "{} {:2.0f}% \n".format(
                class_names[k], 100 * third
            )
        # write the most possible label below the image
        st.image(image[t])
        st.write("Predicted: {}".format(prediction))


def multi_label_roc(
    output_path,
    y_true,
    y_pred,
    labels_list=[
        "Normal",
        "Diabetes",
        "Glaucoma",
        "Cataract",
        "AMD",
        "Hypertension",
        "Myopia",
    ],
):
    """multi_label_roc plot roc for each label

    Parameters
    ----------
    output_path : str
        path to save plot there (should include the name)
    y_true : list or numpy.ndarray
        actual labels
    y_pred : list or numpy.ndarray
        predicted labels
    labels_list : list, optional
        list of the label will be plot, by default ['Normal', 'Diabetes', 'Glaucoma','Cataract', 'AMD',  'Hypertension', 'Myopia',]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = (y_pred.argmax(1)[:, None] == np.arange(y_pred.shape[1])).astype(int)

    for cls in range(len(labels_list)):
        fpr, tpr, _ = roc_curve(y_true[:, cls], y_pred[:, cls])

        plt.plot(
            100 * fpr,
            100 * tpr,
            label=labels_list[cls],
            linewidth=2,
        )
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend(loc="lower right")
    plt.savefig(output_path + ".png")
    # plt.show()
    plt.close()


def model_compare(x_test, y_test, models, tags, output_path, metrics, loss):
    """model_compare compares models in a graph based on given metrics

    Parameters
    ----------
    x_test : tensor
        testing input as a tensor
    y_test : tensor
        actual labels as a tensor
    models : keras.Model
        models that will be compared
    tags: list[str]
        tag for each model to show in plot, respective to models order
    path : str
        path and name the graph to be saved
    metrics : list, optional
        the metrics for comparing models,
    loss : keras.loss or str
        loss function to evaluate models
    """
    tag_idx = 0
    for model in models:
        calculated_metrics, metrics_names = get_specific_metrics(
            x_test, y_test, model, metrics, loss
        )
        y = calculated_metrics
        x = metrics_names
        plt.plot(x, y, label=tags[tag_idx])
        tag_idx += 1
        labels = metrics_names
        plt.xticks(x, labels, rotation="vertical")
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
    plt.legend()
    # plt.show()
    plt.savefig(output_path + ".png")
