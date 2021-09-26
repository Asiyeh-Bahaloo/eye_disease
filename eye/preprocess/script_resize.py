import os, sys
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from preprocess import resize_image


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
    parser.add_argument(
        "-iw",
        "--image_width",
        type=int,
        help="Width of the image you want to resize",
        required=True,
        default="/Dataset/preprocessed",
    )
    parser.add_argument(
        "-k",
        "--keep_aspect_ration",
        type=bool,
        help="Do we keep the original image hight and width ratio or just resize to a square?",
        required=True,
        default="/Dataset/preprocessed",
    )

    args = parser.parse_args()

    resize_image(
        args.image_width, args.source_dir, args.target_dir, args.keep_aspect_ration
    )


## Running
# python script.py -s /Dataset/ToPreprocess -t /Dataset/resized_images -iw 224 -k False
