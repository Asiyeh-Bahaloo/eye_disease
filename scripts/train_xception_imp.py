import os
import argparse
import mlflow
import numpy as np
import tensorflow as tf
import warnings
from distutils.util import strtobool
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    CosineDecay,
    InverseTimeDecay,
)

from eye.models.xception_imp import Xception
from eye.utils import plotter_utils as p
from eye.utils.utils import MlflowCallback, split_ODIR, add_args_to_mlflow
from eye.data.dataloader import ODIR_Dataloader
from eye.data.dataset import ODIR_Dataset
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

from eye.evaluation.metrics import (
    final_score,
    kappa_score,
    loss_per_class,
    accuracy_per_class,
    precision_per_class,
    recall_per_class,
    kappa_per_class,
    f1_per_class,
    auc_per_class,
    final_per_class,
    specificity_per_class,
    sensitivity_per_class,
    micro_auc,
    micro_recall,
    micro_precision,
    micro_specificity,
    micro_sensitivity,
    micro_f1_score,
    accuracy_score,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Xception model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="number of batchs you want to have in your training process",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs you want to train your model",
        required=False,
    )
    parser.add_argument(
        "--pre_epochs",
        type=int,
        default=0,
        help="number of epochs you want to train your final classifier dense layers in transfer learning",
        required=False,
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=5,
        help="number of patience you want to use for early stopping",
        required=False,
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        default="binary_crossentropy",
        help="type of loss function with which you want to compile your model",
        required=False,
    )
    parser.add_argument(
        "--imgnetweights",
        dest="imagenet_weights",
        type=str,
        choices=("True", "False"),
        default="True",
        help="determines to load imagenet pretrained weight or not",
        required=False,
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=str,
        help="Path to the model's pretrained weights file",
        required=False,
    )
    parser.add_argument(
        "--data_train",
        dest="data_folder_train",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )
    parser.add_argument(
        "--data_val",
        dest="data_folder_val",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )
    parser.add_argument(
        "--train_label",
        dest="train_label",
        type=str,
        default="/Data",
        help="Path to the .csv file of train labels.",
        required=True,
    )
    parser.add_argument(
        "--val_label",
        dest="val_label",
        type=str,
        default="/Data",
        help="Path to the .csv file of train labels.",
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
    parser.add_argument(
        "--learning_rate",
        dest="lr_init",
        type=float,
        default=0.001,
        help="setting learning rate of optimizer",
    )
    parser.add_argument(
        "--LR_type",
        dest="lr_type",
        type=str,
        help="Type of the LR scheduler you want to use. It can be 'ED':ExponentialDecay | 'CD': CosineDecay | 'ITD': InverseTimeDecay",
    )
    parser.add_argument(
        "--LR_decay_rate",
        dest="LR_decay",
        type=float,
        default=0.95,
        help="setting decay rate of Learning Rate Schedule.",
    )
    parser.add_argument(
        "--LR_decay_step",
        dest="decay_step",
        type=float,
        default=50,
        help="setting decay step of Learning Rate Schedule.",
    )
    parser.add_argument(
        "--decay_rate",
        dest="decay",
        type=float,
        default=1e-6,
        help="setting decay rate of optimizer",
    )
    parser.add_argument(
        "--momentum_rate",
        dest="momentum",
        type=float,
        default=0.9,
        help="setting momentum rate of optimizer",
    )
    parser.add_argument(
        "--nesterov_flag",
        dest="nesterov",
        type=str,
        choices=("True", "False"),
        default="True",
        help="setting nesterov term of  optimizer: True or False",
    )
    parser.add_argument(
        "--exp",
        dest="experiment",
        type=str,
        default="test-experiment",
        help="setting the experiment under which mlflow must be logged",
    )
    parser.add_argument(
        "--bg_scale",
        dest="bengraham_scale",
        type=int,
        default=350,
        help="setting the scale for BenGraham transform",
    )
    parser.add_argument(
        "--shape",
        dest="shape",
        type=int,
        default=224,
        help="setting shape of image for resize transform",
    )
    parser.add_argument(
        "--keep_AR",
        dest="keepAspectRatio",
        type=str,
        choices=("True", "False"),
        default="True",
        help="whether to keep aspect ratio for resize transform or not",
    )
    parser.add_argument(
        "--tv_frac",
        dest="train_val_fraction",
        type=float,
        default=0.8,
        help="a fraction to split train and validation images",
    )
    parser.add_argument(
        "--data_frac",
        dest="data_fraction",
        type=float,
        default=1,
        help="fraction of all the data we want to train on them",
    )
    parser.add_argument(
        "--ES_mode",
        dest="early_stopping_mode",
        type=str,
        default="min",
        help="setting the mode of early stopping callback",
    )
    parser.add_argument(
        "--ES_monitor",
        dest="early_stopping_monitor",
        type=str,
        default="val_loss",
        help="setting the monitor type of early stopping callback",
    )
    parser.add_argument(
        "--ES_verbose",
        dest="early_stopping_verbose",
        type=int,
        default=1,
        help="setting the verbose of early stopping callback",
    )
    parser.add_argument(
        "--dev_cmt",
        dest="last_Dev_commit",
        type=str,
        help="Last dev commit that you are using for training",
        required=True,
    )
    parser.add_argument(
        "--Auth_name",
        dest="authur_name",
        type=str,
        help="enter your name here plz",
        required=True,
    )
    parser.add_argument(
        "--desc",
        dest="Description",
        type=str,
        help="enter a short description to show what your aim for this run is",
        required=True,
    )
    parser.add_argument(
        "--weight_decay_rate",
        dest="weight_decay_rate",
        type=float,
        default=0.2,
        help="l2 regularization rate",
    )
    parser.add_argument(
        "--dropout_rate",
        dest="dropout_rate",
        type=float,
        default=0.25,
        help="dropout rate of the models",
    )
    args = parser.parse_args()
    return args


# python scripts/train_xception.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data
def main():
    args = parse_arguments()
    tf.config.run_functions_eagerly(True)
    np.seterr(divide="ignore", invalid="ignore")
    warnings.filterwarnings("ignore")

    # Parameters
    num_classes = 8
    tag = "xception_imp_88_unfreezed"

    mlflow.set_experiment(args.experiment)
    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", tag)

    # create compose for both train and validation sets
    compose = Compose(
        transforms=[
            # RemovePadding(),
            # BenGraham(args.bengraham_scale),
            # Resize((args.shape, args.shape), strtobool(args.keepAspectRatio)),
            KerasPreprocess(model_name="xception"),
            # RandomShift(0.2, 0.3),
            # RandomFlipLR(),
            # RandomFlipUD(),
        ]
    )

    # create both train and validation sets
    train_dataset = ODIR_Dataset(
        img_folder_path=args.data_folder_train,
        csv_path=args.train_label,
        img_shape=(args.shape, args.shape),
        num_classes=8,
        frac=args.data_fraction,
        transforms=compose,
    )

    val_dataset = ODIR_Dataset(
        img_folder_path=args.data_folder_val,
        csv_path=args.val_label,
        img_shape=(args.shape, args.shape),
        num_classes=8,
        frac=args.data_fraction,
        transforms=compose,
    )

    # create both train and validation dataloaders
    train_DL = ODIR_Dataloader(dataset=train_dataset, batch_size=args.batch_size)
    val_DL = ODIR_Dataloader(dataset=val_dataset, batch_size=args.batch_size)

    # Model
    model = Xception(
        num_classes=num_classes,
        input_shape=(args.shape, args.shape, 3),
        dropout_rate=args.dropout_rate,
        weight_decay_rate=args.weight_decay_rate,
    )
    if strtobool(args.imagenet_weights):
        model.load_imagenet_weights()
        print("Imagenet weights loaded")
    if args.weights_path is not None:
        model.load_weights(path=args.weights_path)
        print("Weights loaded from the path given")

    add_args_to_mlflow(args)
    mlflow.log_param("Training data size", len(train_dataset))
    mlflow.log_param("Validation data size", len(val_dataset))

    # Set Schedules for LR
    if args.lr_type in ["ED", "CD", "ITD"]:

        if args.lr_type == "ED":
            LR_schedule = ExponentialDecay(
                initial_learning_rate=args.lr_init,
                decay_steps=args.decay_step,
                decay_rate=args.LR_decay,
            )
        elif args.lr_type == "CD":
            LR_schedule = CosineDecay(
                initial_learning_rate=args.lr_init, decay_steps=args.decay_step
            )
        elif args.lr_type == "ITD":
            LR_schedule = InverseTimeDecay(
                initial_learning_rate=args.lr_init,
                decay_steps=args.decay_step,
                decay_rate=args.LR_decay,
            )
        sgd = SGD(
            learning_rate=LR_schedule,
            momentum=args.momentum,
            nesterov=strtobool(args.nesterov),
        )
    else:

        sgd = SGD(
            learning_rate=args.lr_init,
            decay=args.decay,
            momentum=args.momentum,
            nesterov=strtobool(args.nesterov),
        )

    # Metrics
    metrics = [
        accuracy_score,
        micro_auc,
        final_score,
        kappa_score,
        micro_recall,
        micro_precision,
        micro_specificity,
        micro_sensitivity,
        micro_f1_score,
    ]

    for l in range(num_classes):
        metrics.append(loss_per_class(label=l))
        metrics.append(accuracy_per_class(label=l))
        metrics.append(precision_per_class(label=l))
        metrics.append(recall_per_class(label=l))
        metrics.append(kappa_per_class(label=l))
        metrics.append(f1_per_class(label=l))
        metrics.append(auc_per_class(label=l))
        metrics.append(final_per_class(label=l))
        metrics.append(specificity_per_class(label=l))
        # metrics.append(sensitivity_per_class(label=l))

    # Callbacks
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min", verbose=1
    )

    model_file = os.path.join(args.result, f"model_weights_{tag}.h5")
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        model_file,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode="min",
        monitor="val_loss",
    )

    mlfCallback = MlflowCallback(metrics, sgd)

    if args.pre_epochs > 0:
        history = model.train(
            epochs=args.pre_epochs,
            loss=args.loss,
            metrics=metrics,
            callbacks=[mlfCallback, earlyStoppingCallback, modelCheckpoint],
            optimizer=sgd,
            freeze_backbone=True,
            last_freeze_num=88,
            train_data_loader=train_DL,
            validation_data_loader=val_DL,
            batch_size=args.batch_size,
            steps_per_epoch=len(train_DL),
            shuffle=True,
        )
        print("model trained successfuly(Backbone freezed).")

        # model trainable and non-trainable parameters
        trainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.model.trainable_weights]
        )
        nonTrainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.model.non_trainable_weights]
        )
        totalParams = trainableParams + nonTrainableParams

        mlflow.log_param("Total params Model freezed", totalParams)
        mlflow.log_param("Trainable params Model freezed", trainableParams)
        mlflow.log_param("Non-trainable params Model freezed", nonTrainableParams)

    if args.epochs > 0:
        history = model.train(
            epochs=args.epochs,
            loss=args.loss,
            metrics=metrics,
            callbacks=[mlfCallback, earlyStoppingCallback, modelCheckpoint],
            optimizer=sgd,
            freeze_backbone=False,
            train_data_loader=train_DL,
            validation_data_loader=val_DL,
            batch_size=args.batch_size,
            steps_per_epoch=len(train_DL),
            shuffle=True,
        )
        print("model trained successfuly.")

        # model trainable and non-trainable parameters
        trainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.model.trainable_weights]
        )
        nonTrainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.model.non_trainable_weights]
        )
        totalParams = trainableParams + nonTrainableParams

        mlflow.log_param("Total params Model NOT freezed", totalParams)
        mlflow.log_param("Trainable params Model NOT freezed", trainableParams)
        mlflow.log_param("Non-trainable params Model NOT freezed", nonTrainableParams)

    # Save
    mlflow.log_artifact(model_file)
    print(f"Saved model weights in {model_file}")

    # Set a batch of tags
    tags = {"output_path": args.result, "model_name": tag}
    mlflow.set_tags(tags)

    # Plot
    print("plotting...")
    accuracy_plot_figure = os.path.join(args.result, f"{tag}_accuracy.png")
    metrics_plot_figure = os.path.join(args.result, f"{tag}_metrics.png")
    p.plot_accuracy(history=history, path=accuracy_plot_figure)
    p.plot_metrics(history=history, path=metrics_plot_figure)

    mlflow.log_artifact(accuracy_plot_figure)
    mlflow.log_artifact(metrics_plot_figure)
    mlflow.end_run()


if __name__ == "__main__":
    main()
