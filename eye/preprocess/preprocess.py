# from __future__ import absolutes_import, division, print_function, unicode_literals
import logging
import PIL
import os
from PIL import Image
import numpy as np
import logging.config
from os import listdir
from os.path import isfile, join
import tensorflow as tf

# from skimage import exposure
from absl import app
import matplotlib.pyplot as plt

import cv2
import glob
import numpy

plt.style.use("dark_background")


def resize_image(image_width, source_folder, destination_folder, keep_aspect_ration):
    count = 0
    for f in glob.glob(source_folder + "/*"):
        count += 1
        img = Image.open(f)
        img.show()
        if keep_aspect_ration:
            # it will have the exact same width-to-height ratio as the original photo
            width_percentage = image_width / float(img.size[0])
            height_size = int((float(img.size[1]) * float(width_percentage)))
            img = img.resize((image_width, height_size), PIL.Image.ANTIALIAS)
        else:
            # This will force the image to be square
            img = img.resize((image_width, image_width), PIL.Image.ANTIALIAS)
        if "right" in f:
            # self.logger.debug("Right eye image found. Flipping it")
            img.transpose(Image.FLIP_LEFT_RIGHT).save(
                destination_folder + "/" + str(count) + ".jpg",
                optimize=True,
                quality=100,
            )
        else:
            img.save(
                destination_folder + "/" + str(count) + ".jpg",
                optimize=True,
                quality=100,
            )

        # self.logger.debug("Image saved")


def ben_graham(source_folder, destination_folder, scale):
    # Scale 300 seems to be sufficient; 500 and 1000 may be overkill
    count = 0
    for f in glob.glob(source_folder + "/*"):
        print("loading", f)
        count += 1
        # try:
        img = cv2.imread(f)
        x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        img = cv2.resize(img, (0, 0), fx=s, fy=s)

        b = numpy.zeros(img.shape)
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
        # cv2.imwrite(str(scale) + "_" + f, aa)
        cv2.imwrite(destination_folder + "/" + str(count) + ".jpg", aa)
        # except:
        #     print(f)


def crop_image(source_folder, destination_folder):
    count = 0
    for f in glob.glob(source_folder + "/*"):
        print("loading", f)
        count += 1
        image = cv2.imread(f)

        # Mask of coloured pixels.
        mask = image > 0

        # Coordinates of coloured pixels.
        coordinates = np.argwhere(mask)

        # Binding box of non-black pixels.
        x0, y0, s0 = coordinates.min(axis=0)
        x1, y1, s1 = coordinates.max(axis=0) + 1  # slices are exclusive at the top

        # Get the contents of the bounding box.
        cropped = image[x0:x1, y0:y1]
        # overwrite the same file
        cv2.imwrite(destination_folder + "/" + str(count) + ".jpg", cropped)
