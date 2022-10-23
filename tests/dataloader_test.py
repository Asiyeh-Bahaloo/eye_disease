import os
import pytest
from eye.utils.utils import split_ODIR, split_Cataract
from eye.data.dataset import ODIR_Dataset, Cataract_Dataset
from eye.data.dataloader import ODIR_Dataloader, Cataract_Dataloader, Mix_Dataloader
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


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 8])
def test_ODIR_dataloader_batch(batch_size):
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
        frac=0.003,
        transforms=compose,
    )
    data_set = ODIR_dataset.subset(df_train)

    DL = ODIR_Dataloader(dataset=data_set, batch_size=batch_size)

    for i in range(len(DL)):
        x, y = DL.__iter__()
        assert (
            0 < x.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of images"
        assert (
            0 < y.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of labels"
        assert (
            x.shape[0] == y.shape[0]
        ), f"In Batch number{i} X and Y doesn't have the same number of elements"


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 8])
def test_Cataract_dataloader_batch(batch_size):
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
        frac=0.01,
        transforms=compose,
    )
    data_set = Cataract_dataset.subset(df_train)

    DL = Cataract_Dataloader(dataset=data_set, batch_size=batch_size)

    for i in range(len(DL)):
        x, y = DL.__iter__()
        assert (
            0 < x.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of images"
        assert (
            0 < y.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of labels"
        assert (
            x.shape[0] == y.shape[0]
        ), f"In Batch number{i} X and Y doesn't have the same number of elements"


@pytest.mark.parametrize("batch_size", [4, 6, 8, 16])
def test_MIX_dataloader_batch(batch_size):

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
        frac=0.003,
        transforms=compose,
    )
    data_set_1 = ODIR_dataset.subset(df_train)

    (df_train, df_val) = split_Cataract(Image_path=img_path_2, train_val_frac=0.8)

    # create both train and validation sets
    Cataract_dataset = Cataract_Dataset(
        img_folder_path=img_path_2,
        img_shape=(224, 224),
        num_classes=4,
        frac=0.01,
        transforms=compose,
    )
    data_set_2 = Cataract_dataset.subset(df_train)

    DL = Mix_Dataloader(
        dataset1=data_set_1, dataset2=data_set_2, batch_size=batch_size, frac=0.7
    )

    for i in range(len(DL)):
        x, y = DL.__iter__()
        assert (
            0 < x.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of images"
        assert (
            0 < y.shape[0] <= batch_size
        ), f"Batch number{i} has wrong number of labels"
        assert (
            x.shape[0] == y.shape[0]
        ), f"In Batch number{i} X and Y doesn't have the same number of elements"
