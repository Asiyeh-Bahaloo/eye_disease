import numpy as np
import math
from tensorflow.keras.utils import Sequence


class ODIR_Dataloader(Sequence):
    """
    This is our Dataloader class which gets a dataset object and feeds batches to our model while training.
    """

    def __init__(self, dataset, batch_size):
        """
        Initializer of the class

        Parameters
        ----------
        dataset : ODIR_Dataset object
            The object of our dataset class.
        batch_size : int
            batch size you want to use in the dataloader.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # counter to count numbert of passed iterations in next function
        self.counter = 0

    def __len__(self):
        """
        returns total number of batches

        Returns
        -------
        int
            total number of batches.
        """
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        """
        Gets batch index and returns one batch of data

        Parameters
        ----------
        index : int
            index of the wanted batch.

        Returns
        -------
        numpy.ndarray
            It returns two numpy arrays (X, Y)
        """
        if (index + 1) * self.batch_size > len(self.dataset) - 1:
            batch_image_id = self.dataset.df["ID"][
                index * self.batch_size : len(self.dataset)
            ]

            # X = [self.dataset.get_image(file_id) for file_id in batch_image_id]
            # Y = [self.dataset.get_label(file_id) for file_id in batch_image_id]

            batch_items = [self.dataset.get_item(file_id) for file_id in batch_image_id]
            X = [ls[0] for ls in batch_items]
            Y = [ls[1] for ls in batch_items]

            img_shape = (
                (len(self.dataset) - index * self.batch_size,)
                + self.dataset.img_shape
                + (3,)
            )

            X = np.stack(X, axis=0).reshape(img_shape)
            Y = np.stack(Y, axis=0).reshape(
                (
                    len(self.dataset) - index * self.batch_size,
                    self.dataset.num_classes,
                )
            )
        else:
            batch_image_id = self.dataset.df["ID"][
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            # X = [self.dataset.get_image(file_id) for file_id in batch_image_id]
            # Y = [self.dataset.get_label(file_id) for file_id in batch_image_id]

            batch_items = [self.dataset.get_item(file_id) for file_id in batch_image_id]
            X = [ls[0] for ls in batch_items]
            Y = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size,) + self.dataset.img_shape + (3,)

            X = np.stack(X, axis=0).reshape(img_shape)
            Y = np.stack(Y, axis=0).reshape((self.batch_size, self.dataset.num_classes))

        return X, Y

    def on_epoch_end(self):
        """
        procedures done after one epoch, in this case we shuffle the dataset every epoch.
        """
        self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        """
        This function gives one batch of our data every time you call it.

        Returns
        -------
        numpy.ndarray
            It returns two numpy arrays (X, Y)
        """

        X, Y = self[self.counter]
        self.counter += 1

        if self.counter >= len(self):
            self.on_epoch_end()
            self.counter = 0

        return X, Y


class Cataract_Dataloader(Sequence):
    """
    This is our Dataloader class which gets a dataset object and feeds batches to our model while training.
    """

    def __init__(self, dataset, batch_size):
        """
        Initializer of the class

        Parameters
        ----------
        dataset : ODIR_Dataset object
            The object of our dataset class.
        batch_size : int
            batch size you want to use in the dataloader.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # counter to count numbert of passed iterations in next function
        self.counter = 0

    def __len__(self):
        """
        returns total number of batches

        Returns
        -------
        int
            total number of batches.
        """
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        """
        Gets batch index and returns one batch of data

        Parameters
        ----------
        index : int
            index of the wanted batch.

        Returns
        -------
        numpy.ndarray
            It returns two numpy arrays (X, Y_converted)
            Y_converted is the converted labels to the main dataset we have.
        """
        if (index + 1) * self.batch_size > len(self.dataset) - 1:
            batch_image_id = self.dataset.df["ID"][
                index * self.batch_size : len(self.dataset)
            ]

            # X = [self.dataset.get_image(file_id) for file_id in batch_image_id]
            # Y = [self.dataset.get_label(file_id) for file_id in batch_image_id]

            batch_items = [self.dataset.get_item(file_id) for file_id in batch_image_id]
            X = [ls[0] for ls in batch_items]
            Y = [ls[1] for ls in batch_items]

            img_shape = (
                (len(self.dataset) - index * self.batch_size,)
                + self.dataset.img_shape
                + (3,)
            )

            X = np.stack(X, axis=0).reshape(img_shape)
            Y = np.stack(Y, axis=0).reshape(
                (
                    len(self.dataset) - index * self.batch_size,
                    self.dataset.num_classes,
                )
            )
        else:
            batch_image_id = self.dataset.df["ID"][
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            # X = [self.dataset.get_image(file_id) for file_id in batch_image_id]
            # Y = [self.dataset.get_label(file_id) for file_id in batch_image_id]

            batch_items = [self.dataset.get_item(file_id) for file_id in batch_image_id]
            X = [ls[0] for ls in batch_items]
            Y = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size,) + self.dataset.img_shape + (3,)

            X = np.stack(X, axis=0).reshape(img_shape)
            Y = np.stack(Y, axis=0).reshape((self.batch_size, self.dataset.num_classes))

        Y_converted = np.empty((Y.shape[0], 8), dtype=np.int32)
        for i in range(Y.shape[0]):
            if Y[i][0] == 1:
                # normal
                Y_converted[i] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
            elif Y[i][1] == 1:
                # cataract
                Y_converted[i] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif Y[i][2] == 1:
                # glaucoma
                Y_converted[i] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif Y[i][3] == 1:
                # others
                Y_converted[i] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        Y_converted = Y_converted.reshape((Y.shape[0], 8))

        return X, Y_converted

    def on_epoch_end(self):
        """
        procedures done after one epoch, in this case we shuffle the dataset every epoch.
        """
        self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        """
        This function gives one batch of our data every time you call it.

        Returns
        -------
        numpy.ndarray
            It returns two numpy arrays (X, Y)
        """

        X, Y = self[self.counter]
        self.counter += 1

        if self.counter >= len(self):
            self.on_epoch_end()
            self.counter = 0

        return X, Y
