import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


class Knn:
    def __init__(self, num_neighbors, algorithm):

        """Initialization of knn model"""

        self.num_neighbors = num_neighbors
        self.algorithm = algorithm
        self.model = self.build(self.num_neighbors, algorithm=self.algorithm)

    def build(self, num_neighbors=5, algorithm="auto"):

        """builds the knn model with 5 neighbors and auto algorithm by default

        Parameters
        ----------
        num_neighbors : int
            number of neighbors that model will have
        algorithm : str, optional
            Algorithm used to compute the nearest neighbors, default='auto'

        Returns
        -------
        model [KNeighborsClassifier]
            KNN model
        """

        model = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm=algorithm)
        return model

    def train(
        self,
        train_data_loader=None,
        X=None,
        Y=None,
    ):

        """train : trains the model by input data

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        Y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        cls [KNeighborsClassifier]
            The fitted k-nearest neighbors classifier.
        """

        if train_data_loader == None:
            cls = self.model.fit(X, Y)

        else:
            cls = self.model.fit(
                train_data_loader,
            )

        return cls

    def predictions(self, img_path, model, N=8):

        """Making predictions for the query images

        Parameters
        ----------
        img_path : str
            path to the input image
        model : self.model
            Knn model
        N : int
            Number of images to be returned

        Returns
        -------
        neigh_dist: [ndarray of shape (n_queries, n_neighbors)]
            Array representing the lengths to points, only present if return_distance=True.
        """

        img = image.load_img(img_path)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature = np.array(img_data).flatten().reshape(1, -1)
        res = model.kneighbors(
            feature.reshape(1, -1), return_distance=True, n_neighbors=N
        )
        return res
