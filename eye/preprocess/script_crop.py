import os, sys
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from preprocess import crop_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        help="the address of your data which you want to preprocess them",
        required=True,
        default="/Dataset/To be preprocessed",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        help="Address of the directory which you want to save preprocessed images",
        required=True,
        default="/Dataset/preprocessed",
    )

    args = parser.parse_args()

    crop_image(args.source_dir, args.target_dir)


## Running
# python script_crop.py -s /Dataset/ToPreprocess -t /Dataset/cropped_images
