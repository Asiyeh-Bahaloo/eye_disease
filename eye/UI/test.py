import sys
import os
import numpy as np
from PIL import Image
import streamlit as st

from eye.utils.plotter_utils import plot_image_in_UI


def app(model):
    """app this function used for running test page of ui

    in the main app when you want to diaplay test page you must call this function

    Parameters
    ----------
    model : model object
        the trained model object that is used for diagnosing the disease of fundus image
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

    l = []

    inP = None
    img_file_buffer = st.file_uploader(
        "Upload a PNG image", type="png", accept_multiple_files=True
    )
    if img_file_buffer is not None:
        for i in range(len(img_file_buffer)):
            image = Image.open(img_file_buffer[i])
            img_array = np.array(image)
            l.append(img_array)
            inP = np.array(l)

        if inP is not None:
            y = model.predict(inP)
            plot_image_in_UI(y, l, class_names)
