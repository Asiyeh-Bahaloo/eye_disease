import os
import PIL
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def resize_image(img, image_shape, keep_aspect_ration=False):
    """
    This function will resize an input image

    Parameters
    ----------
    image_shape : tuple
        Width and Height of the input image given in a tuple.
    img : str
        Path to the input image
    keep_aspect_ration : bool
        True: Resize image with respect to the original image hight and width ratio
        False: Just resize to a square
    Returns
    -------
    numpy.ndarray
        returns the image
    """
    img = Image.fromarray((img * 255).astype(np.uint8))
    if keep_aspect_ration:
        # it will have the exact same width-to-height ratio as the original photo
        width_percentage = image_shape[0] / float(img.size[0])
        height_size = int((float(img.size[1]) * float(width_percentage)))
        img = img.resize((image_shape[0], height_size), PIL.Image.ANTIALIAS)
    else:
        # This will force the image to be square
        img = img.resize(image_shape, PIL.Image.ANTIALIAS)
    return np.array(img)


def ben_graham(img, scale):

    """
    The method of the winner of Kaggle competition

    Parameters
    ----------
    img : np.ndarray
        numpy array of the input image

    scale : int
    Scale 300 seems to be sufficient; 500 and 1000 may be overkill

    Returns
    -------
    np.ndarray
        Outputs the image
    """
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    img = cv2.resize(img, (0, 0), fx=s, fy=s)

    b = np.zeros(img.shape)
    cv2.circle(
        b,
        (int(img.shape[1] / 2), int(img.shape[0] / 2)),
        int(scale * 0.9),
        (1, 1, 1),
        -1,
        8,
        0,
    )
    aa = cv2.addWeighted(
        img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128
    ) * b + 128 * (1 - b)

    return aa


def remove_padding(img):
    """
    This function will crop the black parts of the input image

    Parameters
    ----------
    img : np.ndarray
        numpy array of the input image

    Returns
    -------
    np.ndarray
        Outputs the image
    """
    # Mask of coloured pixels.
    mask = img > 0

    # Coordinates of coloured pixels.
    coordinates = np.argwhere(mask)

    # Binding box of non-black pixels.
    x0, y0, s0 = coordinates.min(axis=0)
    x1, y1, s1 = coordinates.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = img[x0:x1, y0:y1]

    return cropped
