import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from eye.models.knn import Knn
from eye.utils.utils import load_data, find_similar_images


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Resnet_v2 model"
    )
    parser.add_argument(
        "--num_neigh",
        type=int,
        dest="n_neighbors",
        default=5,
        help="Number of neighbors to use",
        required=True,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        dest="n_images",
        default=8,
        help="Number of images to be returned",
        required=True,
    )
    parser.add_argument(
        "--algo",
        dest="algorithm",
        type=str,
        default="auto",
        help="Algorithm used to compute the nearest neighbors",
        required=False,
    )
    parser.add_argument(
        "--data_npy",
        dest="data_folder_npy",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )
    parser.add_argument(
        "--data_jpg",
        dest="data_folder_jpg",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )
    parser.add_argument(
        "--img_path",
        dest="img_path",
        type=str,
        default="/Data",
        help="Path to image which you want to get similar images to it",
        required=True,
    )
    parser.add_argument(
        "--result",
        dest="result",
        type=str,
        default="/Data",
        help="Path to the folder you want to save the model's results",
        required=True,
    )
    args = parser.parse_args()
    return args


# python eye/scripts/train_resnet_v2.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data
def main():
    args = parse_arguments()

    # Load data
    # TODO: use dataloaders instead
    (X_train, y_train), (_, _), (X_test, _) = load_data(args.data_folder_npy)

    # Model
    model = Knn(num_neighbors=args.n_neighbors, algorithm=args.algorithm)

    X_train = X_train.reshape(X_train.shape[0], -1)
    # Train
    cls = model.train(
        X=X_train,
        Y=y_train,
    )
    print("model trained successfuly.")

    _, indices = model.predictions(args.img_path, model.model, args.n_images)

    plt.imshow(mpimg.imread(args.img_path), interpolation="lanczos")
    plt.xlabel(args.img_path.split(".")[0] + "_Original Image", fontsize=20)
    plt.show()
    print("Predictions ***********")
    filenames = [s for s in glob.glob(args.data_folder_jpg + "/*")]
    find_similar_images(indices[0], filenames, args.result)
    print("similar images saved successfuly.")


if __name__ == "__main__":
    main()


## How to run?
## python train_knn.py --data_npy /Data/data_npy --data_jpg /Data/data_jpg --num_neigh 9 --num_images 5 --img_path /Data/4678_left.jpg --result /Data/result
