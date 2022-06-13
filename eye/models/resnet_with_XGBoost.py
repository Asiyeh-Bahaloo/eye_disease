import os, sys
import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import VersionAwareLayers
from keras import layers
from keras import backend
from tqdm import tqdm
from keras.models import Model

# from tensorflow.keras.models import Model
from keras.engine import training
from keras.utils import data_utils

from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.multioutput import ClassifierChain

from .model import KerasClsBaseModel


BASE_WEIGHT_URL = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/inception_resnet_v2/"
)


class InceptionResNetV2(KerasClsBaseModel):
    def __init__(self, num_classes, input_shape):
        """Initialization of Resnet_v2 class which clarify model architecture"""

        # super().__init__(num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model, self.model_XGB = self.build(self.num_classes, self.input_shape)
        self.pop_flag = True

    def conv2d_bn(
        self,
        x,
        filters,
        kernel_size,
        dropout,
        dropout_rate=0.25,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=False,
        name=None,
    ):
        """Utility function to apply conv + BN.
        Args:
            x: input tensor.
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            strides: strides in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            use_bias: whether to use a bias in `Conv2D`.
            name: name of the ops; will become `name + '_ac'` for the activation
                and `name + '_bn'` for the batch norm layer.
        Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name,
        )(x)
        if dropout:
            x = Dropout(dropout_rate)(x, training=True)
        if not use_bias:
            bn_axis = 1 if backend.image_data_format() == "channels_first" else 3
            bn_name = None if name is None else name + "_bn"
            x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + "_ac"
            x = layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(
        self, x, scale, block_type, block_idx, activation="relu"
    ):
        """Adds an Inception-ResNet block.
        This function builds 3 types of Inception-ResNet blocks mentioned
        in the paper, controlled by the `block_type` argument (which is the
        block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
        Args:
            x: input tensor.
            scale: scaling factor to scale the residuals (i.e., the output of passing
            `x` through an inception module) before adding them to the shortcut
            branch. Let `r` be the output from the residual branch, the output of this
            block will be `x + scale * r`.
            block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
            structure in the residual branch.
            block_idx: an `int` used for generating layer names. The Inception-ResNet
            blocks are repeated many times in this network. We use `block_idx` to
            identify each of the repetitions. For example, the first
            Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
            and the layer names will have a common prefix `'block35_0'`.
            activation: activation function to use at the end of the block (see
            [activations](../activations.md)). When `activation=None`, no activation
            is applied
            (i.e., "linear" activation: `a(x) = x`).
        Returns:
            Output tensor for the block.
        Raises:
            ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
        """
        if block_type == "block35":
            branch_0 = self.conv2d_bn(x, 32, 1, dropout=False, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(x, 32, 1, dropout=False, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(branch_1, 32, 3, dropout=False, dropout_rate=0.25)
            branch_2 = self.conv2d_bn(x, 32, 1, dropout=False, dropout_rate=0.25)
            branch_2 = self.conv2d_bn(branch_2, 48, 3, dropout=False, dropout_rate=0.25)
            branch_2 = self.conv2d_bn(branch_2, 64, 3, dropout=False, dropout_rate=0.25)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == "block17":
            branch_0 = self.conv2d_bn(x, 192, 1, dropout=False, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(x, 128, 1, dropout=False, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(
                branch_1, 160, [1, 7], dropout=False, dropout_rate=0.25
            )
            branch_1 = self.conv2d_bn(
                branch_1, 192, [7, 1], dropout=False, dropout_rate=0.25
            )
            branches = [branch_0, branch_1]
        elif block_type == "block8":
            branch_0 = self.conv2d_bn(x, 192, 1, dropout=True, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(x, 192, 1, dropout=True, dropout_rate=0.25)
            branch_1 = self.conv2d_bn(
                branch_1, 224, [1, 3], dropout=True, dropout_rate=0.25
            )
            branch_1 = self.conv2d_bn(
                branch_1, 256, [3, 1], dropout=True, dropout_rate=0.25
            )
            branches = [branch_0, branch_1]
        else:
            raise ValueError(
                "Unknown Inception-ResNet block type. "
                'Expects "block35", "block17" or "block8", '
                "but got: " + str(block_type)
            )

        block_name = block_type + "_" + str(block_idx)
        channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
        mixed = layers.Concatenate(axis=channel_axis, name=block_name + "_mixed")(
            branches
        )
        up = self.conv2d_bn(
            mixed,
            backend.int_shape(x)[channel_axis],
            1,
            dropout=False,
            dropout_rate=0.25,
            activation=None,
            use_bias=True,
            name=block_name + "_conv",
        )

        x = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=backend.int_shape(x)[1:],
            arguments={"scale": scale},
            name=block_name,
        )([x, up])
        if activation is not None:
            x = layers.Activation(activation, name=block_name + "_ac")(x)
        return x

    def build(self, num_classes, input_shape):
        """builds the model architecture by default uses random weights

        Parameters
        ----------
        num_classes : int
            number of classes that model should detect
        input_shape :

        Returns
        -------
        model [tensorflow.keras.models.Model]
              model's Architecture
        """

        img_input = layers.Input(shape=input_shape)

        x = self.conv2d_bn(
            img_input,
            32,
            3,
            dropout=False,
            dropout_rate=0.25,
            strides=2,
            padding="valid",
        )
        x = self.conv2d_bn(x, 32, 3, dropout=False, dropout_rate=0.25, padding="valid")
        x = self.conv2d_bn(x, 64, 3, dropout=False, dropout_rate=0.25)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = self.conv2d_bn(x, 80, 1, dropout=False, dropout_rate=0.25, padding="valid")
        x = self.conv2d_bn(x, 192, 3, dropout=False, dropout_rate=0.25, padding="valid")
        x = layers.MaxPooling2D(3, strides=2)(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 320
        branch_0 = self.conv2d_bn(x, 96, 1, dropout=False, dropout_rate=0.25)
        branch_1 = self.conv2d_bn(x, 48, 1, dropout=False, dropout_rate=0.25)
        branch_1 = self.conv2d_bn(branch_1, 64, 5, dropout=False, dropout_rate=0.25)
        branch_2 = self.conv2d_bn(x, 64, 1, dropout=False, dropout_rate=0.25)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, dropout=False, dropout_rate=0.25)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, dropout=False, dropout_rate=0.25)
        branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
        branch_pool = self.conv2d_bn(
            branch_pool, 64, 1, dropout=False, dropout_rate=0.25
        )
        branches = [branch_0, branch_1, branch_2, branch_pool]
        channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
        x = layers.Concatenate(axis=channel_axis, name="mixed_5b")(branches)

        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = self.inception_resnet_block(
                x, scale=0.17, block_type="block35", block_idx=block_idx
            )

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        branch_0 = self.conv2d_bn(
            x, 384, 3, dropout=False, dropout_rate=0.25, strides=2, padding="valid"
        )
        branch_1 = self.conv2d_bn(x, 256, 1, dropout=False, dropout_rate=0.25)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, dropout=False, dropout_rate=0.25)
        branch_1 = self.conv2d_bn(
            branch_1,
            384,
            3,
            dropout=False,
            dropout_rate=0.25,
            strides=2,
            padding="valid",
        )
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name="mixed_6a")(branches)

        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = self.inception_resnet_block(
                x, scale=0.1, block_type="block17", block_idx=block_idx
            )

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = self.conv2d_bn(x, 256, 1, dropout=True, dropout_rate=0.25)
        branch_0 = self.conv2d_bn(
            branch_0,
            384,
            3,
            dropout=True,
            dropout_rate=0.25,
            strides=2,
            padding="valid",
        )
        branch_1 = self.conv2d_bn(x, 256, 1, dropout=True, dropout_rate=0.25)
        branch_1 = self.conv2d_bn(
            branch_1,
            288,
            3,
            dropout=True,
            dropout_rate=0.25,
            strides=2,
            padding="valid",
        )
        branch_2 = self.conv2d_bn(x, 256, 1, dropout=True, dropout_rate=0.25)
        branch_2 = self.conv2d_bn(branch_2, 288, 3, dropout=True, dropout_rate=0.25)
        branch_2 = self.conv2d_bn(
            branch_2,
            320,
            3,
            dropout=True,
            dropout_rate=0.25,
            strides=2,
            padding="valid",
        )
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name="mixed_7a")(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = self.inception_resnet_block(
                x, scale=0.2, block_type="block8", block_idx=block_idx
            )
        x = self.inception_resnet_block(
            x, scale=1.0, activation=None, block_type="block8", block_idx=10
        )

        # Final convolution block: 8 x 8 x 1536
        x = self.conv2d_bn(x, 1536, 1, dropout=True, dropout_rate=0.25, name="conv_7b")

        ## top part
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(num_classes, activation="sigmoid")(x)

        inputs = img_input
        # Create model.
        model = training.Model(inputs, x, name="inception_resnet_v2")
        # model = Model(inputs=backbone.input, outputs=x)

        # creat XGB layer
        XGB = XGBClassifier()
        model_XGB = ClassifierChain(XGB, order="random", random_state=0)

        return model, model_XGB

    def load_imagenet_weights(self):
        """loads imagenet-pretrained weight into model"""

        ### remember to pop layersssss !!!!!!!!!!!!!!!
        ### I have to pop last 2 layers and load imagenet weights

        x = self.model.layers[-3].output
        self.model = training.Model(self.model.input, x)

        fname = "inception_resnet_v2_weights_" "tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = data_utils.get_file(
            fname,
            BASE_WEIGHT_URL + fname,
            cache_subdir="models",
            file_hash="d19885ff4a710c122648d3b5c3b684e4",
        )
        self.model.load_weights(weights_path)

        x = self.model.layers[-1].output
        x = Dense(1024, activation="relu")(x)
        x = Dense(self.num_classes, activation="sigmoid")(x)
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

        # remove last layer of net work
        if self.pop_flag:
            self.model = Model(
                inputs=self.model.input, outputs=self.model.layers[-2].output
            )

        feat_list = []
        y_list = []
        feat_val_list = []
        y_val_list = []

        # get each batch from data loader
        for i in tqdm(range(len(train_data_loader))):
            x, y = train_data_loader.__iter__()

            feat = self.model.predict(x)
            feat_list.append(feat)
            y_list.append(y)

        feat_from_cnn = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        for i in tqdm(range(len(validation_data_loader))):
            x, y = validation_data_loader.__iter__()

            feat = self.model.predict(x)
            feat_val_list.append(feat)
            y_val_list.append(y)

        feat_val_from_cnn = np.concatenate(feat_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        print(
            f"y_train shape is:{y_train.shape}, features shape is: {feat_from_cnn.shape}"
        )

        self.model_XGB.fit(feat_from_cnn, y_train)
        print("XGB layer trained successfully")

        pred_train_y = self.model_XGB.predict_proba(feat_from_cnn)
        pred_val_y = self.model_XGB.predict_proba(feat_val_from_cnn)

        pred_train_y = np.array(pred_train_y)
        pred_val_y = np.array(pred_val_y)

        training_result = {}
        validation_result = {}
        training_result_per_class = {}
        validation_result_per_class = {}

        print(f"y_pred shape= {pred_train_y.shape}")
        for i in range(9):
            name = metrics[i].__name__
            train_value = metrics[i](y_train, pred_train_y)
            training_result[name] = train_value

            validation_value = metrics[i](y_val, pred_val_y)
            validation_result[name] = validation_value

        # convert numpy array to tf.tensor for useing pre class metrics function
        tensor_y_train = tf.convert_to_tensor(y_train, np.float32)
        tensor_pred_train_y = tf.convert_to_tensor(pred_train_y, np.float32)
        tensor_y_val = tf.convert_to_tensor(y_val, np.float32)
        tensor_pred_val_y = tf.convert_to_tensor(pred_val_y, np.float32)

        # Per class metrics
        for i in range(9, len(metrics)):
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

    def load_weights(self, path, gboost_path=None, **kwargs):
        """load_weights loads the weights of pretrained CNN & XGB model

        Parameters
        ----------
        path : srt
            path of CNN model file (a .h5 file)
        gboost_path : str
            path of XGB model file (a .sav file)

        Returns
        -------
        keras & sklearn model object
            the CNN and SVM model as output
        """

        if gboost_path != None:
            self.model_XGB = pickle.load(open(gboost_path, "rb"))

        self.model.load_weights(path, **kwargs), self.model_XGB

    def predict(self, img, **kwargs):
        """predict gets an image as a tensor and predict the label

        Parameters
        ----------
        img : tensor
            input image as a tensor
        """

        # remove last layer of net work
        if self.pop_flag:
            self.model = Model(
                inputs=self.model.input, outputs=self.model.layers[-2].output
            )

        # extract features fron VGG16
        feat = self.model.predict(img)
        # get final output
        pred = self.model_XGB.predict_proba(feat)

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
            self.model = Model(
                inputs=self.model.input, outputs=self.model.layers[-2].output
            )

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

        pred = self.model_XGB.predict_proba(
            test_feat
        )  # pred value with svm for training dataset

        testing_result = {}
        testing_result_per_class = {}

        for i in range(9):
            name = metrics[i].__name__

            test_value = metrics[i](y_test, pred)
            testing_result[name] = test_value

        tensor_y_test = tf.convert_to_tensor(y_test, np.float32)
        tensor_pred_test_y = tf.convert_to_tensor(pred, np.float32)

        # Per class metrics
        for i in range(9, len(metrics)):
            name = metrics[i].__name__

            test_value = metrics[i](tensor_y_test, tensor_pred_test_y)
            testing_result_per_class[name] = test_value

        return (testing_result, testing_result_per_class)
