from matplotlib.pyplot import axis
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras.losses import BinaryCrossentropy

from .model import KerasClsBaseModel

from eye.evaluation.metrics import (
    final_score,
    kappa_score,
    loss_per_class,
    accuracy_per_class,
    precision_per_class,
    recall_per_class,
    kappa_per_class,
    f1_per_class,
    auc_per_class,
    final_per_class,
    specificity_per_class,
    sensitivity_per_class,
    micro_auc,
    micro_recall,
    micro_precision,
    micro_specificity,
    micro_sensitivity,
    micro_f1_score,
    accuracy_score,
    loss,
)


class InceptionV3_Xgboost(KerasClsBaseModel):
    def __init__(self, num_classes):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """
        self.num_classes = num_classes
        self.model, self.model_XGB = self.build(self.num_classes)

    def build(self, num_classes, pretrained_backbone=None):
        """builds the model architecture by default uses random weights

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        pretrained_backbone : tensorflow.keras.models.Model, optional
            contains backbone model with pretrained weights, by default None

        Returns
        -------
        model [tensorflow.keras.models.Model]
              model's Architecture
        """

        if pretrained_backbone is not None:
            backbone = pretrained_backbone
        else:
            backbone = inception_v3.InceptionV3(weights=None, include_top=False)

        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=backbone.input, outputs=x)

        XGB = XGBClassifier()
        model_XGB = MultiOutputClassifier(XGB)

        return model, model_XGB

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """
        backbone = inception_v3.InceptionV3(weights="imagenet", include_top=False)
        self.model, self.model_XGB = self.build(self.num_classes, backbone)

    def train(
        self,
        epochs,
        loss,
        metrics,
        callbacks=...,
        optimizer=...,
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
        # train model like other model

        history = 0

        if not freeze_backbone:
            history = super().train(
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

        # pop the last layer
        # print(f"weights1= {self.model.layers[-2].get_weights()}")
        model2 = Model(self.model.input, self.model.layers[-2].output)
        # print(f"weights2= {model2.layers[-1].get_weights()}")

        feat_list = []
        y_list = []
        feat_val_list = []
        y_val_list = []

        # get each batch from data loader
        for i in tqdm(range(len(train_data_loader))):
            x, y = train_data_loader.__iter__()

            # feat = self.model.predict(x)
            feat = model2.predict(x)
            feat_list.append(feat)
            y_list.append(y)

        feat_from_cnn = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        for i in tqdm(range(len(validation_data_loader))):
            x, y = validation_data_loader.__iter__()

            # feat = self.model.predict(x)
            feat = model2.predict(x)
            feat_val_list.append(feat)
            y_val_list.append(y)

        feat_val_from_cnn = np.concatenate(feat_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        ## Feed features extracted from Inception_V3 model to XGBoost classifier

        print(
            f"y_train shape is:{y_train.shape}, features shape is: {feat_from_cnn.shape}"
        )
        # y_train shape is:(52, 8), features shape is: (52, 100)
        # clf = self.model_XGB.fit(feat_from_cnn, y_train)
        self.model_XGB.fit(feat_from_cnn, y_train)

        pred_train_y = self.model_XGB.predict(feat_from_cnn)
        pred_val_y = self.model_XGB.predict(feat_val_from_cnn)
        # pred_train_y = clf.predict(feat_from_cnn)
        # pred_val_y = clf.predict(feat_val_from_cnn)

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

        # # Accuracy
        # # metrics[0].update_state(y_train, pred_train_y)
        # # training_result["accuracy"] = metrics[0].result().numpy()
        # # metrics[0].update_state(y_val, pred_val_y)
        # # validation_result["accuracy"] = metrics[0].result().numpy()
        # training_result["accuracy"] = accuracy_score(y_train, pred_train_y)
        # validation_result["accuracy"] = accuracy_score(y_val, pred_val_y)

        # # Precision
        # # metrics[1].update_state(y_train, pred_train_y)
        # # training_result["precision"] = metrics[1].result().numpy()
        # # metrics[1].update_state(y_val, pred_val_y)
        # # validation_result["precision"] = metrics[1].result().numpy()
        # # metrics[1].update_state(y_train, pred_train_y)
        # training_result["precision"] = micro_precision(y_train, pred_train_y)
        # # metrics[1].update_state(y_val, pred_val_y)
        # validation_result["precision"] = micro_precision(y_val, pred_val_y)

        # # Recall
        # # metrics[2].update_state(y_train, pred_train_y)
        # # training_result["recall"] = metrics[2].result().numpy()
        # # metrics[2].update_state(y_val, pred_val_y)
        # # validation_result["recall"] = metrics[2].result().numpy()
        # # metrics[2].update_state(y_train, pred_train_y)
        # training_result["recall"] = micro_recall(y_train, pred_train_y)
        # # metrics[2].update_state(y_val, pred_val_y)
        # validation_result["recall"] = micro_recall(y_val, pred_val_y)

        # # AUC
        # # metrics[3].update_state(y_train, pred_train_y)
        # # training_result["auc"] = metrics[3].result().numpy()
        # # metrics[3].update_state(y_val, pred_val_y)
        # # validation_result["auc"] = metrics[3].result().numpy()
        # # metrics[3].update_state(y_train, pred_train_y)
        # training_result["auc"] = micro_auc(y_train, pred_train_y)
        # # metrics[3].update_state(y_val, pred_val_y)
        # validation_result["auc"] = micro_auc(y_val, pred_val_y)

        # # Sensitivity
        # # training_result["sensitivity"] = metrics[4](y_train, pred_train_y).numpy()
        # # validation_result["sensitivity"] = metrics[4](y_val, pred_val_y).numpy()
        # training_result["sensitivity"] = micro_sensitivity(y_train, pred_train_y)
        # validation_result["sensitivity"] = micro_sensitivity(y_val, pred_val_y)

        # # Specificity
        # ## eager tensor!
        # # training_result["specificity"] = metrics[5](y_train, pred_train_y).numpy()
        # # validation_result["specificity"] = metrics[5](y_val, pred_val_y).numpy()
        # training_result["specificity"] = micro_specificity(y_train, pred_train_y)
        # validation_result["specificity"] = micro_specificity(y_val, pred_val_y)

        # convert numpy array to tf.tensor for useing pre class metrics function
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

    def save(self, path, xgboost_path, **kwargs):
        """save function save model

        this function save svm layer and CNN net work

        Parameters
        ----------
        path : str
            path of file you wnat to save CNN model (.h5 file)
        xgboost_path : str
            path of file you want to save svm model (.pickle.dat file)
        """
        super().save(path, **kwargs)
        # save XGboost model to file
        pickle.dump(self.model_XGB, open(xgboost_path, "wb"))
