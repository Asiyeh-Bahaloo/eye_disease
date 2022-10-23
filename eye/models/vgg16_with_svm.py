import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.multioutput import ClassifierChain
from sklearn import svm
from tensorflow.python.keras.backend import l2_normalize
import numpy as np
import pickle

from .model import KerasClsBaseModel


class Vgg16(KerasClsBaseModel):
    """Vgg16 a model based on VGG16 model to detect disease

    Parameters
    ----------
    num_classes : int
        number of classes of the classification task
    """

    def __init__(self, num_classes, input_shape, C, kernel):
        """__init__ set number of classes and builds model architecture

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        """
        # super().__init__(num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.C = C
        self.kernel = kernel
        self.model, self.svm_layer = self.build(
            self.num_classes, self.input_shape, self.C, self.kernel
        )
        self.pop_flag = True

    def build(self, num_classes, input_shape, C, kernel):
        """builds the model architecture by default uses random weights

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect

        Returns
        -------
        model [keras sequential model]
              model's Architecture
        """

        # create svm model with indicating kernel type(kernel) and regularization parameter(C)
        svm_layer = svm.SVC(C=C, kernel=kernel, probability=True)
        # make the output of svm model a multi label out put
        chain = ClassifierChain(svm_layer, order="random", random_state=0)

        trainable = False
        last_part_trainable = True
        model = models.Sequential()
        # Block 1
        layer = layers.Conv2D(
            input_shape=input_shape,
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 2
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 3
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 4
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        model.add(layer)

        # Block 5
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = trainable
        model.add(layer)
        layer = layers.Conv2D(
            512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        layer.trainable = last_part_trainable
        model.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = last_part_trainable
        model.add(layer)

        layer = layers.Flatten()
        layer.trainable = last_part_trainable
        model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = last_part_trainable
        model.add(layer)
        layer = layers.Dense(4096, activation="relu")
        layer.trainable = last_part_trainable
        model.add(layer)
        layer = layers.Dense(10, activation="relu")
        layer.trainable = last_part_trainable
        model.add(layer)
        layer = layers.Dense(num_classes, activation="sigmoid")
        model.add(layer)

        return model, chain

    def load_imagenet_weights(self, path):
        """loads imagenet-pretrained weight into model"""

        self.model.pop()
        self.model.pop()
        layer = layers.Dense(1000, activation="softmax")
        layer.trainable = False
        self.model.add(layer)
        self.model.load_weights(path)

        # Remove last layer
        self.model.pop()

        # Add new dense layer
        self.model.add(layers.Dense(50, activation="relu"))
        self.model.add(layers.Dense(8, activation="sigmoid"))

    def train(
        self,
        epochs,
        loss,
        metrics,
        callbacks=...,
        optimizer=...,
        train_data_loader=None,
        validation_data_loader=None,
        X=None,
        Y=None,
        X_val=None,
        Y_val=None,
        batch_size=32,
        **kwargs
    ):
        """train : trains the model by input data

        Parameters
        ----------
        epochs : int
            number of epochs that training process will repeat
        loss : str or keras.loss
            loss function to calculate gradients
        callbacks : list, optional
            functions that will execute after each eochs, by default []
        optimizer : keras.optimizer, optional
            optimizer method to optimize learning process, by default Adam()
        train_data_loader : keras data generator, optional
            data loader to give training data sectional, by default None
        validation_data_loader : [type], optional
             data loader to give validation data sectional, by default None
        X : tensor, optional
            input data as a tensor, by default None
        Y : tensor, optional
            labels as a tensor, by default None
        X_val : tensor, optional
            validation input as a tensor, by default None
        Y_val : tensor, optional
            validation labels as a tensor, by default None
        batch_size : int, optional
            number of inputs per each calculation, by default 32
        metrics : list or keras metrics, optional
            metrics to describe model performance, by default ...

        """
        # train model like other model
        history = super().train(
            epochs,
            loss,
            metrics,
            callbacks=callbacks,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            X=X,
            Y=Y,
            X_val=X_val,
            Y_val=Y_val,
            batch_size=batch_size,
            **kwargs
        )

        # remove last layer of net work
        self.model.pop()
        self.pop_flag = False

        feat_list = []
        y_list = []
        # get each bath from data loader
        for i in range(len(train_data_loader)):
            x, y = train_data_loader.__iter__()

            # extract features fron VGG16
            feat = self.model.predict(x)
            feat_list.append(feat)
            # save y corresponding to the features
            y_list.append(y)

        feat_from_cnn = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        # fit svm model on train data
        self.svm_layer.fit(feat_from_cnn, y_train)

        # extract validation data from validation data loader
        feat_val_list = []
        y_val_list = []
        for i in range(len(validation_data_loader)):
            x, y = validation_data_loader.__iter__()

            # extract features fron VGG16
            feat = self.model.predict(x)
            feat_val_list.append(feat)
            # save y corresponding to the features
            y_val_list.append(y)

        feat_val_from_cnn = np.concatenate(feat_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        pred_val_y = self.svm_layer.predict(
            feat_val_from_cnn
        )  # pred value with svm for valdation dataset
        pred_train_y = self.svm_layer.predict(
            feat_from_cnn
        )  # pred value with svm for training dataset

        metr_6_train = {}
        metr_6_val = {}

        # calculate 6 metrics :accuracy,precision,recall,auc,sensitivity,specificity

        metrics[0].update_state(y_train, pred_train_y)
        r = metrics[0].result().numpy()
        metr_6_train.update({"accuracy": r})

        # calculate accuracy for validation dataset
        metrics[0].update_state(y_val, pred_val_y)
        r = metrics[0].result().numpy()
        metr_6_val.update({"accuracy": r})
        ###################################################

        # calculate precision for training dataset
        metrics[1].update_state(y_train, pred_train_y)
        r = metrics[1].result().numpy()
        metr_6_train.update({"precision": r})

        # calculate precision for validation dataset
        metrics[1].update_state(y_val, pred_val_y)
        r = metrics[1].result().numpy()
        metr_6_val.update({"precision": r})
        ###################################################

        # calculate recall for training dataset
        metrics[2].update_state(y_train, pred_train_y)
        r = metrics[2].result().numpy()
        metr_6_train.update({"recall": r})

        # calculate recall for validation dataset
        metrics[2].update_state(y_val, pred_val_y)
        r = metrics[2].result().numpy()
        metr_6_val.update({"recall": r})
        ###################################################

        # calculate auc for training dataset
        metrics[3].update_state(y_train, pred_train_y)
        r = metrics[3].result().numpy()
        metr_6_train.update({"auc": r})

        # calculate auc for validation dataset
        metrics[3].update_state(y_val, pred_val_y)
        r = metrics[3].result().numpy()
        metr_6_val.update({"auc": r})
        ###################################################

        # calculate sensitivity for training dataset
        r = metrics[4](y_train, pred_train_y)
        metr_6_train.update({"sensitivity": r})

        # calculate sensitivity for validation dataset
        r = metrics[4](y_val, pred_val_y)
        metr_6_val.update({"sensitivity": r})

        ###################################################

        # calculate specificity for training dataset
        r = metrics[5](y_train, pred_train_y)
        metr_6_train.update({"specificity": r})

        # calculate specificity for validation dataset
        r = metrics[5](y_val, pred_val_y)
        metr_6_val.update({"specificity": r})

        # convert numpy array to tf.tensor for useing pre class metrics function
        tensor_y_train = tf.convert_to_tensor(y_train, np.float32)
        tensor_pred_train_y = tf.convert_to_tensor(pred_train_y, np.float32)
        tensor_y_val = tf.convert_to_tensor(y_val, np.float32)
        tensor_pred_val_y = tf.convert_to_tensor(pred_val_y, np.float32)

        # calculate pre class metrics
        pre_class_value_val = {}
        pre_class_value_train = {}

        for i in range(6, len(metrics)):

            name = metrics[i].__name__
            tm = metrics[i](tensor_y_train, tensor_pred_train_y)
            pre_class_value_train.update({name: tm})

            tm_val = metrics[i](tensor_y_val, tensor_pred_val_y)
            pre_class_value_val.update({name: tm_val})

        return (
            history,
            metr_6_val,
            metr_6_train,
            pre_class_value_val,
            pre_class_value_train,
        )

    def save(self, path, svm_path, **kwargs):
        """save function save model

        this function save svm layer and CNN net work

        Parameters
        ----------
        path : str
            path of file you wnat to save CNN model (.h5 file)
        svm_path : str
            path of file you want to save svm model (.sav file)
        """

        super().save(path, **kwargs)
        pickle.dump(self.svm_layer, open(svm_path, "wb"))

    def load_weights(self, path, path_svm, **kwargs):
        """load_weights loads the weights of pretrained CNN & SVM model

        Parameters
        ----------
        path : srt
            path of CNN model file (a .h5 file)
        path_svm : str
            path of SVM model file (a .sav file)

        Returns
        -------
        keras & sklearn model object
            the CNN and SVM model as output
        """

        self.svm_layer = pickle.load(open(path_svm, "rb"))
        return self.model.load_weights(path, **kwargs), self.svm_layer

    def predict(self, img, **kwargs):
        """predict gets an image as a tensor and predict the label

        Parameters
        ----------
        img : tensor
            input image as a tensor
        """

        if self.pop_flag:
            self.model.pop()

        # extract features fron VGG16
        feat = self.model.predict(img)
        # get final output
        pred = self.svm_layer.predict(feat)

        return pred

    def evaluate(self, metrics, loss, test_data_loader=None, X=None, Y=None, **kwargs):
        """evaluate a method to evaluate model performance

        Parameters
        ----------
        loss : keras.loss
            loss function to calculate gradients
        test_data_loader : keras data generator, optional
            data loader to give testing data sectional, by default None
        X : tensor, optional
            testing input as a tensor, by default None
        Y : tensor, optional
            testing labels as a tensor, by default None
        metrics : list, optional
            list of metrics to desctibe model performance, by default ["accuracy"]
        """
        if self.pop_flag:
            self.model.pop()

        self.model.compile(metrics=metrics, loss=loss)

        if test_data_loader == None:
            test_feat = self.model.predict(X)
            y_test = Y
        else:
            feat_list = []
            y_list = []
            # get each bath from data loader
            for i in range(len(test_data_loader)):
                x, y = test_data_loader.__iter__()

                # extract features fron VGG16
                feat = self.model.predict(x)
                feat_list.append(feat)
                # save y corresponding to the features
                y_list.append(y)

            test_feat = np.concatenate(feat_list, axis=0)
            y_test = np.concatenate(y_list, axis=0)

        pred = self.svm_layer.predict(
            test_feat
        )  # pred value with svm for training dataset

        metr_6_test = {}

        # calculate 6 metrics :accuracy,precision,recall,auc,sensitivity,specificity

        metrics[0].update_state(y_test, pred)
        r = metrics[0].result().numpy()
        metr_6_test.update({"accuracy": r})
        ###################################################

        # calculate precision for training dataset
        metrics[1].update_state(y_test, pred)
        r = metrics[1].result().numpy()
        metr_6_test.update({"precision": r})
        ###################################################

        # calculate recall for training dataset
        metrics[2].update_state(y_test, pred)
        r = metrics[2].result().numpy()
        metr_6_test.update({"recall": r})
        ###################################################

        # calculate auc for training dataset
        metrics[3].update_state(y_test, pred)
        r = metrics[3].result().numpy()
        metr_6_test.update({"auc": r})
        ###################################################

        # calculate sensitivity for training dataset
        r = metrics[4](y_test, pred)
        metr_6_test.update({"sensitivity": r})
        ###################################################

        # calculate specificity for training dataset
        r = metrics[5](y_test, pred)
        metr_6_test.update({"specificity": r})
        ###################################################

        # convert numpy array to tf.tensor for useing pre class metrics function
        tensor_y_test = tf.convert_to_tensor(y_test, np.float32)
        tensor_pred = tf.convert_to_tensor(pred, np.float32)

        # calculate pre class metrics
        pre_class_value_test = {}

        for i in range(6, len(metrics)):

            name = metrics[i].__name__
            tm = metrics[i](tensor_y_test, tensor_pred)
            pre_class_value_test.update({name: tm})

        return (
            metr_6_test,
            pre_class_value_test,
        )
