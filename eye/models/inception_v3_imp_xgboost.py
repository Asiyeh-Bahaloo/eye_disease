from matplotlib.pyplot import axis
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
from keras import backend
from keras.engine import training
import h5py

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import layers
from keras.utils import data_utils

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


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


def load_attributes_from_hdf5_group(group, name):

    """Loads attributes of the specified name from the HDF5 group.
    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.
    Returns:
        data: Attributes data.
    """
    if name in group.attrs:
        data = [
            n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]
        ]
    else:
        data = []
        chunk_id = 0
        while "%s%d" % (name, chunk_id) in group.attrs:
            data.extend(
                [
                    n.decode("utf8") if hasattr(n, "decode") else n
                    for n in group.attrs["%s%d" % (name, chunk_id)]
                ]
            )
            chunk_id += 1
    return data


def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.
    Args:
        f: A pointer to a HDF5 group.
    Returns:
        List of NumPy arrays of the weight values.
    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")
    return [np.asarray(f[weight_name]) for weight_name in weight_names]


class InceptionV3_Xgboost(KerasClsBaseModel):
    def __init__(self, num_classes, input_shape):
        """__init__  is  constructor function of inception class

        this class creat a customized Inception_v3 model for classifying task

        Parameters
        ----------

        num_classes : int
            number of classes of task
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model, self.model_XGB = self.build(self.num_classes, self.input_shape)

    def conv2d_bn(
        self,
        x,
        filters,
        num_row,
        num_col,
        padding="same",
        strides=(1, 1),
        name=None,
        dropout_rate=None,
    ):
        """Utility function to apply conv + BN.
        Args:
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        bn_axis = 3
        x = layers.Conv2D(
            filters,
            (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        )(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x, training=True)
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = layers.Activation("relu", name=name)(x)
        return x

    def build(self, num_classes, input_shape):
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

        img_input = layers.Input(shape=input_shape)
        channel_axis = 3  ## channels_last data format

        x = self.conv2d_bn(
            img_input, 32, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )
        x = self.conv2d_bn(x, 32, 3, 3, padding="valid", dropout_rate=0.25)
        x = self.conv2d_bn(x, 64, 3, 3, dropout_rate=0.25)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding="valid", dropout_rate=0.25)
        x = self.conv2d_bn(x, 192, 3, 3, padding="valid", dropout_rate=0.25)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=0.25)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=0.25)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed0",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=0.25)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=0.25)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed1",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 2: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, dropout_rate=0.25)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, dropout_rate=0.25)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed2",
        )([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(
            x, 384, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, dropout_rate=0.25)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed3")(
            [branch3x3, branch3x3dbl, branch_pool]
        )

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=0.25)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed4",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1, dropout_rate=0.25)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7, dropout_rate=0.25)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=0.25)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1, dropout_rate=0.25)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1, dropout_rate=0.25)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7, dropout_rate=0.25)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1, dropout_rate=0.25)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=0.25)
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(5 + i),
            )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7, dropout_rate=0.25)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, dropout_rate=0.25)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, dropout_rate=0.25)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, dropout_rate=0.25)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=0.25)
        x = layers.Concatenate(
            axis=channel_axis,
            name="mixed7",
        )([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch3x3 = self.conv2d_bn(
            branch3x3, 320, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1, dropout_rate=0.25)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding="valid", dropout_rate=0.25
        )

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.Concatenate(axis=channel_axis, name="mixed8")(
            [branch3x3, branch7x7x3, branch_pool]
        )

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1, dropout_rate=0.25)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1, dropout_rate=0.25)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3, dropout_rate=0.25)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1, dropout_rate=0.25)
            branch3x3 = layers.Concatenate(axis=channel_axis, name="mixed9_" + str(i))(
                [branch3x3_1, branch3x3_2]
            )

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1, dropout_rate=0.25)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3, dropout_rate=0.25)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3, dropout_rate=0.25)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1, dropout_rate=0.25)
            branch3x3dbl = layers.Concatenate(axis=channel_axis)(
                [branch3x3dbl_1, branch3x3dbl_2]
            )

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, dropout_rate=0.25)
            x = layers.Concatenate(
                axis=channel_axis,
                name="mixed" + str(9 + i),
            )([branch1x1, branch3x3, branch3x3dbl, branch_pool])

        inputs = img_input
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)

        model = training.Model(inputs, x, name="inception_v3")

        XGB = XGBClassifier()
        model_XGB = MultiOutputClassifier(XGB)

        return model, model_XGB

    def load_imagenet_weights(self):
        """load_imagenet_weights function used for loading pretrained weight of task imagenet

        Here we only load the weights of the convolutional part
        """

        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        weights_path = data_utils.get_file(
            "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="bcbd6486424b2319ff4ef7d526e38f63",
        )

        with h5py.File(weights_path, "r") as f:
            # changing model layer names
            for layer in self.model.layers:
                # print(f"$$$$$$$$before: {layer.name}")
                if layer.name in ["input_1", "mixed9_0", "mixed9_1"]:
                    continue
                if layer.name in [
                    "conv2d",
                    "batch_normalization",
                    "activation",
                    "max_pooling2d",
                    "average_pooling2d",
                    "dropout",
                    "concatenate",
                ]:
                    layer._name = layer.name + "_1"
                else:
                    idx = layer.name.rfind("_")
                    if idx != -1:
                        try:
                            layer._name = layer.name[: idx + 1] + str(
                                int(layer.name[idx + 1 :]) + 1
                            )
                            # print(f"$$$$$$$$AFTERR: {layer.name}")
                        except:
                            continue

            # model_layer_name = []
            # for layer in self.model.layers:
            #   model_layer_name.append(layer.name)

            file_layer_names = load_attributes_from_hdf5_group(f, "layer_names")

            # print("########model:")
            # print(sorted(model_layer_name))
            # print("########file:")
            # print(file_layer_names)

            for layer_name in file_layer_names:
                if (layer_name != "average_pooling2d_10") and (
                    "batch_normalization" not in layer_name
                ):
                    # print(layer_name)
                    layer = self.model.get_layer(layer_name)
                    weight_values = load_subset_weights_from_hdf5_group(f[layer_name])
                    layer.set_weights(weight_values)

        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(self.num_classes, activation="sigmoid")(x)

        self.model = training.Model(self.model.input, x)

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
