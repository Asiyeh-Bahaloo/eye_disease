from abc import ABC

from tensorflow.keras.optimizers import Adam


class BaseModel(ABC):
    def __init__(self):
        pass

    def build(self):
        raise NotImplementedError()

    def load_imagenet_weights(self):
        raise NotImplementedError()

    def train(
        self,
        metrics,
        callbacks,
        epochs,
        loss,
        optimizer,
        train_data_loader=None,
        validation_data_loader=None,
        X=None,
        Y=None,
        X_val=None,
        Y_val=None,
        batch_size=32,
        **kwargs,
    ):
        raise NotImplementedError()

    def evaluate(
        self,
        loss,
        metrics,
        test_data_loader=None,
        X=None,
        Y=None,
        **kwargs,
    ):
        raise NotImplementedError()

    def predict(self, img, **kwargs):
        raise NotImplementedError()

    def save(self, path, **kwargs):
        raise NotImplementedError()

    def load_weights(self, path, **kwargs):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()


class KerasClsBaseModel(BaseModel):
    """A Base Model for Keras classification models.

    Parameters
    ----------
    num_class : int
        number of classes of the classification task.
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.model = self.build(self.num_class)

    def build(self, num_class, pretrained_backbone=None):
        """[summary]

        Parameters
        ----------
        num_class : int
            number of classes of the classification task
        pretrained_backbone : tensorflow.keras.models.Model, optional
            [description], by default None

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError()

    def train(
        self,
        epochs,
        loss,
        metrics,
        callbacks=[],
        optimizer=Adam(),
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
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if train_data_loader == None:
            history = self.model.fit(
                x=X,
                y=Y,
                callbacks=callbacks,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, Y_val),
                **kwargs,
            )
        else:
            history = self.model.fit(
                x=train_data_loader,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data_loader,
                callbacks=callbacks,
                **kwargs,
            )

        return history

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
        self.model.compile(metrics=metrics, loss=loss)
        if test_data_loader == None:
            baseline_results = self.model.evaluate(
                X,
                Y,
                **kwargs,
            )
        else:
            baseline_results = self.model.evaluate(test_data_loader, **kwargs)
        return baseline_results

    def predict(self, img, **kwargs):
        """predict gets an image as a tensor and predict the label

        Parameters
        ----------
        img : tensor
            input image as a tensor
        """

        return self.model.predict(img, **kwargs)

    def save(self, path, **kwargs):
        """This function saves the weights of the model in a (.h5) file.

        Parameters
        ----------
        path : str
            Direct path to the file you want to save your weights on.
        """
        self.model.save(path, **kwargs)

    def load_weights(self, path, **kwargs):
        """load_weights : loads the weights of pretrained model
            input model should be in same architect that this model has

        Parameters
        ----------
        path : str
            dir of the pretrained model
        """
        return self.model.load_weights(path, **kwargs)

    def summary(self):
        """summary gives a brief information about model and its architecture"""
        self.model.summary()
