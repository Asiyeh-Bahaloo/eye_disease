import numpy as np
from PIL import Image
from numpy.lib.type_check import imag
import streamlit as st
import cv2


from eye.UI.seg_ui_functions import segment, segment_av
from eye.utils.plotter_utils import plot_image_in_UI
from eye.data.preprocess import ben_graham


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

            # segmentation without classifying vessel
            s1 = segment(image)

            # segmentation without classifying vessel
            s2 = segment_av(image)

            img_array = np.array(image)
            l.append(img_array)
            inP = np.array(l)

            # give image to ben graham transform
            ben = ben_graham(img_array, 300)
            t = (ben * 255).astype(np.uint8)
            x = cv2.resize(
                t, (300, 300)
            )  # x :the output of ben graham transform ,prepared for displaying in UI

            if inP is not None:
                y = model.predict(inP)
                # display image with probable labels
                plot_image_in_UI(y, l, class_names)

                # display ben graham image
                st.image(x, caption="BENGRAHAM TRANSFORM")

                # display segmentation image
                st.image([s1, s2])

                l = []
