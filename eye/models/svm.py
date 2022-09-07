import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn import svm
from tensorflow.python.keras.backend import l2_normalize
import numpy as np
import pickle
from tqdm import tqdm


class SVM:
    """a unit class for creat/tune/train/validate/test svm classifier"""

    def __init__(self, backbone, multi_label, num_pop, kernel, C):
        """constructor function of svm class

        Parameters
        ----------
        backbone : keras model
            the CNN model ,we want to extracts features from that,and pass them to the SVM
        multi_label : str
            indicate the ,how we handel the multilabel problem for the svm
        num_pop : int
            number of last layers of model ,that we wnat to omit
        kernel : str
            indicate the kernel type of model
        C : float
            regularization parameter of svm
        """

        self.backbone = backbone
        self.multi_label = multi_label
        self.num_pop = num_pop
        self.kernel = kernel
        self.C = C
        self.model_svm = self.build(
            multi_label=self.multi_label, C=self.C, kernel=self.kernel
        )

    def build(self, multi_label, kernel, C):
        """that create sklearn SVM model,that can handle the multilabel data

        Parameters
        ----------
        multi_label : str
            indicate the ,how we handel the multilabel problem for the svm
        kernel : str
            indicate the kernel type of model
        C : float
            regularization parameter of svm

        Returns
        -------
        sklearn model
            the svm model we want
        """
        # creat sklearn svm(svc for classification) model,that for non multilabel dataset
        svm_ = svm.SVC(C=C, kernel=kernel, probability=True)

        # handel multilable part of model
        if multi_label == "ClassifierChain":
            model_svm = ClassifierChain(svm_, order="random", random_state=0)
        elif multi_label == "MultiOutputClassifier":
            model_svm = MultiOutputClassifier(svm_)
        return model_svm

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """

        self.backbone.load_imagenet_weights()

    def train(
        self,
        epochs,
        loss,
        metrics,
        callbacks=...,
        optimizer=...,
        num_pop=1,
        freeze_backbone=True,
        train_data_loader=None,
        validation_data_loader=None,
        X=None,
        Y=None,
        X_val=None,
        Y_val=None,
        batch_size=32,
        **kwargs,
    ):
        history = 0
        # Train backbone if intended
        if not freeze_backbone:
            history = self.backbone.train(
                epochs,
                loss,
                metrics,
                callbacks=callbacks,
                optimizer=optimizer,
                freeze_backbone=False,
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
                X=X,
                Y=Y,
                X_val=X_val,
                Y_val=Y_val,
                batch_size=batch_size,
                **kwargs,
            )

        # Pop the last layer
        model2 = Model(
            self.backbone.model.input, self.backbone.model.layers[-(num_pop + 1)].output
        )

        feat_list = []
        y_list = []
        feat_val_list = []
        y_val_list = []

        # get each batch from data loader
        for i in tqdm(range(len(train_data_loader))):
            x, y = train_data_loader.__iter__()
            feat = model2.predict(x)
            feat_list.append(feat)
            y_list.append(y)

        feat_from_cnn = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        print("******", feat_from_cnn.shape)
        for i in tqdm(range(len(validation_data_loader))):
            x, y = validation_data_loader.__iter__()
            feat = model2.predict(x)
            feat_val_list.append(feat)
            y_val_list.append(y)

        feat_val_from_cnn = np.concatenate(feat_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        ## Feed features extracted from backbone to XGBoost classifier
        self.model_svm.fit(feat_from_cnn, y_train)

        pred_train_y = self.model_svm.predict(feat_from_cnn)
        pred_val_y = self.model_svm.predict(feat_val_from_cnn)

        training_result = {}
        validation_result = {}
        training_result_per_class = {}
        validation_result_per_class = {}

        print(f"y_pred shape= {pred_train_y.shape}")
        for i in range(10):
            name = metrics[i].__name__

            train_value = metrics[i](y_train, pred_train_y)
            training_result[name] = train_value

            validation_value = metrics[i](y_val, pred_val_y)
            validation_result[name] = validation_value

        tensor_y_train = tf.convert_to_tensor(y_train, np.float32)
        tensor_pred_train_y = tf.convert_to_tensor(pred_train_y, np.float32)
        tensor_y_val = tf.convert_to_tensor(y_val, np.float32)
        tensor_pred_val_y = tf.convert_to_tensor(pred_val_y, np.float32)

        # Per class metrics
        for i in range(10, len(metrics)):
            name = metrics[i].__name__

            train_value = metrics[i](tensor_y_train, tensor_pred_train_y)
            training_result_per_class[name] = train_value

            validation_value = metrics[i](tensor_y_val, tensor_pred_val_y)
            validation_result_per_class[name] = validation_value

        return (
            history,
            training_result,
            validation_result,
            training_result_per_class,
            validation_result_per_class,
        )

    def save(self, path, svm_path, **kwargs):
        """save function save model

        this function save svm layer and CNN net work

        Parameters
        ----------
        path : str
            path of file you wnat to save CNN model (.h5 file)
        svm_path : str
            path of file you want to save svm model (.pickle.dat file)
        """
        self.backbone.save(path, **kwargs)
        # save XGboost model to file
        pickle.dump(self.model_svm, open(svm_path, "wb"))
