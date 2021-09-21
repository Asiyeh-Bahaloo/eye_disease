import tensorflow as tf


def training_model(
    model,
    mertics_list,
    optimizer,
    loss_function,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs,
    batch_size,
    shuffle_flag,
    patience,
):
    """training_model function used for training and compiling model

    in this function we set the properties of training

    Parameters
    ----------
    model : model object
        the model that we want to tarin it
    mertics_list : list
        a list that contains an arbitrary number of keras metrics like:
        [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    optimizer : SGD object
        it object of SGD class that set the properties of optimizing algorithm
    loss_function : string
        that indicates the type of loss function in training
    x_train : numpy array
        the array that contains the input of traing data set
    y_train : numpy array
        the array that contains the output(or labels) of traing data set
    x_val : numpy array
        the array that contains the input of validation data set
    y_val : numpy array
        the array that contains the output(or labels) of validation data set
    epochs : integer
        this indicates epochs of training
    batch_size : integer
        this indicates the batch size of mini-bacth of optimizing algorithm
    shuffle_flag : boolean
        this indicates shuffle or not shuffle the data during training
    patience : integer
        this indicates the patience of callback function that used in training

    Returns
    -------
    model object,history object
        the model object is trained model object and history object is the object that contains
        the information of trainig primary model
    """

    model.compile(loss=loss_function, optimizer=optimizer, metrics=mertics_list)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        validation_data=(x_val, y_val),
        callbacks=[callback],
    )
    return model, history
