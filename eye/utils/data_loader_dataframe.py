from pandas import read_csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_loader(path_to_dataframe, path_to_train_images, bach_size=32):
    """loads data using a dataframe

    :param path_to_dataframe: full path to dataframe contains data
    :param path_to_train_images: directory of the images
    :param bach_size: the size of the returning batchs
    :return: data generator to use in ml models
    """
    data = read_csv(path_to_dataframe)
    label = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    data_loader = ImageDataGenerator()

    data_loader = data_loader.flow_from_dataframe(
        dataframe=data,
        directory=path_to_train_images,
        x_col='ID',
        y_col=label,
        target_size=(224, 224),
        class_mode='raw',
        bach_size=bach_size,
        validate_filenames=False

    )

    return data_loader
