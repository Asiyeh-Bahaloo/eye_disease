from typing import Tuple
import pandas as pd
import os
import cv2
import numpy as np


class ODIR_Dataset:
    """
    This class models the ODIR dataset.
    """

    def __init__(
        self,
        img_folder_path,
        img_shape,
        num_classes,
        frac,
        transforms,
        csv_path=None,
        dataframe=None,
        random_state=2021,
    ):
        """
        Initializer for the ODIR_dataset class.

        Parameters
        ----------
        img_folder_path : str
            Direct path to the FOLDER contating the images.
        csv_path : str
            Direct path to the CSV FILE contating the labels(train, val).
        dataframe : pandas.DataFrame
            Two DataFrames given in a tuple that contains the labels of our images(train, val).
            or just one DataFrame that represents the train or val sets.
        img_shape : tuple
            a 2D tuple that shows the each image shape. (e.g. (224, 224))
        num_classes : int
            total number of classes in dataset.
        frac : float
            Number to set the fraction of data you want to consider.
        transforms: Compose object
            Object of the Compose class that contains all the transform you want to use.
        random_state : int
            random_state used during this function
        """

        self.img_folder_path = img_folder_path
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.frac = frac
        self.transforms = transforms

        if not (csv_path is None):
            self.df = pd.read_csv(csv_path)
            self.df = self.df.sample(frac=frac, random_state=random_state).reset_index(
                drop=True
            )
        elif isinstance(dataframe, pd.DataFrame):
            self.df = dataframe.sample(
                frac=frac, random_state=random_state
            ).reset_index(drop=True)
        else:
            print("Dataframe format isn't correct")

    def subset(self, dataframe):
        """
        This function return object of the ODIR_Dataset related to the set you asked (train set or validation set)

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame that includes the mapping of labels for images.

        Returns
        -------
        ODIR_Dataset object
            ODIR_Dataset object of the asked portion.
        """
        return type(self)(
            img_folder_path=self.img_folder_path,
            dataframe=dataframe,
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            frac=self.frac,
            transforms=self.transforms,
        )

    def __len__(self):
        """
        Returns the length of our dataset. (number of images we have.)

        Returns
        -------
        int
            length of our dataset.
        """
        return len(self.df)

    def get_item(self, image_id):
        """
        It gets image_id then returns the image and its label.

        Parameters
        ----------
        image_id : str
            ID of the image you want. (e.g. 7_right.jpg)

        Returns
        -------
        image, label
            wanted image and its label.
        """

        path = os.path.join(self.img_folder_path, image_id)
        img = cv2.imread(path)
        img = self.transforms(img)

        class_names = [
            "Normal",
            "Diabetes",
            "Glaucoma",
            "Cataract",
            "AMD",
            "Hypertension",
            "Myopia",
            "Others",
        ]
        label = np.asarray(self.df.loc[self.df["ID"] == image_id, class_names])

        return img, label

    def get_image(self, image_id):
        """
        This function gives the image_id then loads the image and do defined preprocesses and finally returns the numpy array of the image.

        Parameters
        ----------
        image_id : str
            ID of the image in dataset (e.g. 7_right.jpg)

        Returns
        -------
        numpy.ndarray
            preprocessed image.
        """

        path = os.path.join(self.img_folder_path, image_id)
        img = cv2.imread(path)
        img = self.transforms(img)
        return img

    def get_label(self, image_id):
        """
        This function reads and returns the label of the given image.

        Parameters
        ----------
        image_id : str
            ID of the image in dataset (e.g. 7_right.jpg)

        Returns
        -------
        numpy.ndarray
            Label of the given image (having shape (#num_classes,))
        """
        class_names = [
            "Normal",
            "Diabetes",
            "Glaucoma",
            "Cataract",
            "AMD",
            "Hypertension",
            "Myopia",
            "Others",
        ]
        label = np.asarray(self.df.loc[self.df["ID"] == image_id, class_names])
        return label


class Cataract_Dataset:
    """
    This class models the Cataract dataset.
    """

    def __init__(
        self,
        img_folder_path,
        img_shape,
        num_classes,
        frac,
        transforms,
        random_state=2021,
    ):
        """
        Initializer for the Cataract_dataset class.

        Parameters
        ----------
        img_folder_path : str
            Direct path to the FOLDER contating the images.
        dataframe : pandas.DataFrame
            Two DataFrames given in a tuple that contains the labels of our images(train, val).
            or just one DataFrame that represents the train or val sets.
        img_shape : tuple
            a 2D tuple that shows the each image shape. (e.g. (224, 224))
        num_classes : int
            total number of classes in dataset.
        frac : float
            Number to set the fraction of data you want to consider.
        transforms: Compose object
            Object of the Compose class that contains all the transform you want to use.
        random_state : int
            random_state used during this function
        """

        self.img_folder_path = img_folder_path
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.frac = frac
        self.transforms = transforms
        self.df = (
            self.get_Cataract_df()
            .sample(frac=self.frac, random_state=random_state)
            .reset_index(drop=True)
        )

    def get_Cataract_df(self, random_state=2021):
        """
        This function makes the label dataframe for Cataract dataset

        Parameters
        ----------
        Image_path : str
            Path to the image folder of the cataract dataset. this folder must have 4 folders in it:
            - 1_normal
            - 2_cataract
            - 2_glaucoma
            - 3_retina_disease
        random_state : int, optional
            random_state used during this function, by default 2021

        Returns
        -------
        DataFrames
            df
        """

        df = pd.DataFrame(columns=["ID", "normal", "cataract", "glaucoma", "others"])

        for idx1, filename in enumerate(
            os.listdir(os.path.join(self.img_folder_path, "1_normal"))
        ):
            df.loc[idx1] = [filename, 1, 0, 0, 0]
        for idx2, filename in enumerate(
            os.listdir(os.path.join(self.img_folder_path, "2_cataract"))
        ):
            df.loc[idx2 + idx1 + 1] = [filename, 0, 1, 0, 0]
        for idx3, filename in enumerate(
            os.listdir(os.path.join(self.img_folder_path, "2_glaucoma"))
        ):
            df.loc[idx3 + idx2 + idx1 + 2] = [filename, 0, 0, 1, 0]

        # shuffle dataset
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        return df

    def subset(self, dataframe):
        """
        This function return object of the Cataract_Dataset related to the set you asked (train set or validation set)

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame that includes the mapping of labels for images.

        Returns
        -------
        Cataract_Dataset object
            Cataract_Dataset object of the asked portion.
        """
        return type(self)(
            img_folder_path=self.img_folder_path,
            dataframe=dataframe,
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            frac=self.frac,
            transforms=self.transforms,
        )

    def __len__(self):
        """
        Returns the length of our dataset. (number of images we have.)

        Returns
        -------
        int
            length of our dataset.
        """
        return len(self.df)

    def get_item(self, image_id):
        """
        It gets image_id then returns the image and its label.

        Parameters
        ----------
        image_id : str
            ID of the image you want. (e.g. 7_right.jpg)

        Returns
        -------
        image, label
            wanted image and its label.
        """
        if image_id.find("NL") > -1:
            path_image_folder = "1_normal"
        elif image_id.find("cataract") > -1:
            path_image_folder = "2_cataract"
        elif image_id.find("Glaucoma") > -1:
            path_image_folder = "2_glaucoma"
        elif image_id.find("Retina") > -1:
            path_image_folder = "3_retina_disease"

        path = os.path.join(self.img_folder_path, path_image_folder, image_id)
        img = cv2.imread(path)
        img = self.transforms(img)

        class_names = [
            "normal",
            "cataract",
            "glaucoma",
            "others",
        ]
        label = np.asarray(self.df.loc[self.df["ID"] == image_id, class_names])

        return img, label

    def get_image(self, image_id):
        """
        This function gives the image_id then loads the image and do defined preprocesses and finally returns the numpy array of the image.

        Parameters
        ----------
        image_id : str
            ID of the image in dataset (e.g. 7_right.jpg)

        Returns
        -------
        numpy.ndarray
            preprocessed image.
        """

        if image_id.find("NL") > -1:
            path_image_folder = "1_normal"
        elif image_id.find("cataract") > -1:
            path_image_folder = "2_cataract"
        elif image_id.find("Glaucoma") > -1:
            path_image_folder = "2_glaucoma"
        elif image_id.find("Retina") > -1:
            path_image_folder = "3_retina_disease"

        path = os.path.join(self.img_folder_path, path_image_folder, image_id)
        img = cv2.imread(path)
        img = self.transforms(img)
        return img

    def get_label(self, image_id):
        """
        This function reads and returns the label of the given image.

        Parameters
        ----------
        image_id : str
            ID of the image in dataset (e.g. 7_right.jpg)

        Returns
        -------
        numpy.ndarray
            Label of the given image (having shape (#num_classes,))
        """
        class_names = [
            "normal",
            "cataract",
            "glaucoma",
            "others",
        ]
        label = np.asarray(self.df.loc[self.df["ID"] == image_id, class_names])
        return label
