import os
import pandas as pd
import pytest
import math
from eye.utils.utils import split_ODIR, split_Cataract
from eye.data.dataset import ODIR_Dataset, Cataract_Dataset
from eye.data.transforms import (
    Compose,
    Resize,
    RemovePadding,
    BenGraham,
    RandomShift,
    RandomFlipLR,
    RandomFlipUD,
    KerasPreprocess,
)

img_path_1 = os.environ.get("IMAGE_FOLDER_1")
img_path_2 = os.environ.get("IMAGE_FOLDER_2")
tv_label = os.environ.get("TRAIN_VAL_LABELS")
test_label = os.environ.get("TEST_LABELS")


@pytest.mark.parametrize("fraction", [1, 0.1, 0.23, 0.3, 0.4, 0.51, 0.6, 0.7, 0.8, 0.9])
def test_odir_dataset_frac(fraction):
    compose = Compose(
        transforms=[
            RemovePadding(),
            BenGraham(350),
            Resize((224, 224), False),
            KerasPreprocess(model_name="vgg16"),
            # RandomShift(0.2, 0.3),
            # RandomFlipLR(),
            # RandomFlipUD(),
        ]
    )

    (df_train, df_val) = split_ODIR(path_train_val=tv_label, train_val_frac=0.8)
    # create both train and validation sets
    ODIR_dataset = ODIR_Dataset(
        img_folder_path=img_path_1,
        csv_path=tv_label,
        img_shape=(224, 224),
        num_classes=8,
        frac=fraction,
        transforms=compose,
    )
    data_set = ODIR_dataset.subset(df_train)

    assert len(data_set) == math.ceil(fraction * len(df_train)) or len(
        data_set
    ) == math.floor(
        fraction * len(df_train)
    ), "fraction doesn't work well with fraction=" + str(
        fraction
    )


@pytest.mark.parametrize("fraction", [1, 0.1, 0.23, 0.3, 0.4, 0.51, 0.6, 0.9])
def test_Cataract_dataset_frac(fraction):

    compose = Compose(
        transforms=[
            RemovePadding(),
            BenGraham(350),
            Resize((224, 224), False),
            KerasPreprocess(model_name="vgg16"),
            # RandomShift(0.2, 0.3),
            # RandomFlipLR(),
            # RandomFlipUD(),
        ]
    )

    (df_train, df_val) = split_Cataract(Image_path=img_path_2, train_val_frac=0.8)
    # create both train and validation sets
    Cataract_dataset = Cataract_Dataset(
        img_folder_path=img_path_2,
        img_shape=(224, 224),
        num_classes=4,
        frac=fraction,
        transforms=compose,
    )
    data_set = Cataract_dataset.subset(df_train)

    assert len(data_set) == math.ceil(fraction * len(df_train)) or len(
        data_set
    ) == math.floor(
        fraction * len(df_train)
    ), "fraction doesn't work well with fraction=" + str(
        fraction
    )
