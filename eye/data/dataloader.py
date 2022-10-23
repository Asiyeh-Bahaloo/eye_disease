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


class Mix_Dataloader(Sequence):
    """
    This is our Dataloader class which gets two dataset objects and feeds batches to our model while training.
    """

    def __init__(self, dataset1, dataset2, batch_size, frac=0.75):
        """
        Initializer of the class

        Parameters
        ----------
        dataset1 : ODIR_Dataset object
            The object of our first dataset class.
        dataset2 : Catarct_dataset object
            The object of our second dataset class.
        batch_size : int
            batch size you want to use in the dataloader.
        frac : float
            fraction you want to have in each batch from dataset1 and dataset 2. (in range(0,1))
            frac = 0.6 means that 0.6 of each batch will contain images from dataset1 and 0.4 of images of dataset2.
        """

        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.frac = frac
        self.batch_size_1 = math.ceil(frac * self.batch_size)
        self.batch_size_2 = self.batch_size - self.batch_size_1

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
        return math.ceil((len(self.dataset1) + len(self.dataset2)) / self.batch_size)

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
        # First data set
        if index * self.batch_size_1 > len(self.dataset1) - 1:
            start = np.random.randint(len(self.dataset1) - self.batch_size_1 - 2)

            batch_image_id = self.dataset1.df["ID"][start : start + self.batch_size_1]

            batch_items = [
                self.dataset1.get_item(file_id) for file_id in batch_image_id
            ]
            X1 = [ls[0] for ls in batch_items]
            Y1 = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size_1,) + self.dataset1.img_shape + (3,)

            X1 = np.stack(X1, axis=0).reshape(img_shape)
            Y1 = np.stack(Y1, axis=0).reshape(
                (self.batch_size_1, self.dataset1.num_classes)
            )
        elif (index + 1) * self.batch_size_1 > len(self.dataset1) - 1:

            batch_image_id = self.dataset1.df["ID"][
                index * self.batch_size_1 : len(self.dataset1)
            ]

            batch_items = [
                self.dataset1.get_item(file_id) for file_id in batch_image_id
            ]
            X1 = [ls[0] for ls in batch_items]
            Y1 = [ls[1] for ls in batch_items]

            img_shape = (
                (len(self.dataset1) - index * self.batch_size_1,)
                + self.dataset1.img_shape
                + (3,)
            )

            X1 = np.stack(X1, axis=0).reshape(img_shape)
            Y1 = np.stack(Y1, axis=0).reshape(
                (
                    len(self.dataset1) - index * self.batch_size_1,
                    self.dataset1.num_classes,
                )
            )
        else:
            batch_image_id = self.dataset1.df["ID"][
                index * self.batch_size_1 : (index + 1) * self.batch_size_1
            ]

            batch_items = [
                self.dataset1.get_item(file_id) for file_id in batch_image_id
            ]
            X1 = [ls[0] for ls in batch_items]
            Y1 = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size_1,) + self.dataset1.img_shape + (3,)

            X1 = np.stack(X1, axis=0).reshape(img_shape)
            Y1 = np.stack(Y1, axis=0).reshape(
                (self.batch_size_1, self.dataset1.num_classes)
            )

        # second data set
        if index * self.batch_size_2 > len(self.dataset2) - 1:
            start = np.random.randint(len(self.dataset2) - self.batch_size_2 - 2)

            batch_image_id = self.dataset2.df["ID"][start : start + self.batch_size_2]

            batch_items = [
                self.dataset2.get_item(file_id) for file_id in batch_image_id
            ]
            X2 = [ls[0] for ls in batch_items]
            Y2 = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size_2,) + self.dataset2.img_shape + (3,)

            X2 = np.stack(X2, axis=0).reshape(img_shape)
            Y2 = np.stack(Y2, axis=0).reshape(
                (self.batch_size_2, self.dataset2.num_classes)
            )

        elif (index + 1) * self.batch_size_2 > len(self.dataset2) - 1:

            batch_image_id = self.dataset2.df["ID"][
                index * self.batch_size_2 : len(self.dataset2)
            ]

            batch_items = [
                self.dataset2.get_item(file_id) for file_id in batch_image_id
            ]
            X2 = [ls[0] for ls in batch_items]
            Y2 = [ls[1] for ls in batch_items]

            img_shape = (
                (len(self.dataset2) - index * self.batch_size_2,)
                + self.dataset2.img_shape
                + (3,)
            )

            X2 = np.stack(X2, axis=0).reshape(img_shape)
            Y2 = np.stack(Y2, axis=0).reshape(
                (
                    len(self.dataset2) - index * self.batch_size_2,
                    self.dataset2.num_classes,
                )
            )
        else:
            batch_image_id = self.dataset2.df["ID"][
                index * self.batch_size_2 : (index + 1) * self.batch_size_2
            ]

            batch_items = [
                self.dataset2.get_item(file_id) for file_id in batch_image_id
            ]
            X2 = [ls[0] for ls in batch_items]
            Y2 = [ls[1] for ls in batch_items]

            img_shape = (self.batch_size_2,) + self.dataset2.img_shape + (3,)

            X2 = np.stack(X2, axis=0).reshape(img_shape)
            Y2 = np.stack(Y2, axis=0).reshape(
                (self.batch_size_2, self.dataset2.num_classes)
            )

        Y2_converted = np.empty((Y2.shape[0], 8), dtype=np.int32)

        for i in range(Y2.shape[0]):
            if Y2[i][0] == 1:
                # normal
                Y2_converted[i] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
            elif Y2[i][1] == 1:
                # cataract
                Y2_converted[i] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif Y2[i][2] == 1:
                # glaucoma
                Y2_converted[i] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif Y2[i][3] == 1:
                # others
                Y2_converted[i] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        Y2_converted = Y2_converted.reshape((Y2.shape[0], 8))

        X = np.concatenate((X1, X2), axis=0)
        Y = np.concatenate((Y1, Y2_converted), axis=0)

        # shuffling data entries in batch
        assert (
            X.shape[0] == Y.shape[0]
        ), "X and Y array don't have the same number of elements, sth is wrong"

        p = np.random.permutation(len(X))

        return X[p], Y[p]

    def on_epoch_end(self):
        """
        procedures done after one epoch, in this case we shuffle the dataset every epoch.
        """
        self.dataset1.df = self.dataset1.df.sample(frac=1).reset_index(drop=True)
        self.dataset2.df = self.dataset2.df.sample(frac=1).reset_index(drop=True)

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
