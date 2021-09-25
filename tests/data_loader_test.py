import numpy
import pytest
import eye.train.Inception_V3.data_loader as data_loader

path_to_dataframe = '/home/khanel/Documents/AI course/Project/s2_p3/odir.csv'
path_to_images = '/home/khanel/Documents/AI course/Project/s2_p3/train/'
bath_size = 32


@pytest.mark.data
def test_data_loader():
    dl = data_loader.data_loader(path_to_dataframe, path_to_images, bath_size)
    assert type(dl[0][0]) == type(numpy.ndarray((1,1,1))) or type(dl[0][0]) == type(numpy.array([]))
    assert len(dl[0]) <= 0

