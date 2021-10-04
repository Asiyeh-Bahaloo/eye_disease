import pandas as pd
import os
import cv2
import numpy as np

from .transforms import Compose


class ODIR_Dataset:
    """
    This class models the ODIR dataset.
    """

    def __init__(
        self,
        img_folder_path,
        csv_path,
        img_shape,
        num_classes,
        frac,
        transforms,
    ):
        """
        Initializer for the ODIR_dataset class.

        Parameters
        ----------
        img_folder_path : str
            Direct path to the FOLDER contating the images.
        csv_path : str
            Direct path to the CSV FILE contating the labels.
        img_shape : tuple
            a 2D tuple that shows the each image shape. (e.g. (224, 224))
        num_classes : int
            total number of classes in dataset.
        frac : float
            Number to set the fraction of data you want to consider.
        do_ben_graham_pre : bool, optional
            Boolian that show if you want to use ben_graham preprocess function, by default False
        do_augmentation : bool, optional
            Boolian that show if you want to use data augmentation, by default False
        """
        self.img_folder_path = img_folder_path
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sample(frac=frac).reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def get_item(self, image_id):

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
        This function reads and returns the given image.

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