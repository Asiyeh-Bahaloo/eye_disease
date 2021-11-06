import os
import argparse
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from eye.models.inception_v3 import InceptionV3
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
    specificity,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
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
        type=bool,
        default=True,
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
        "--data",
        dest="data_folder",
        type=str,
        default="/Data",
        help="Path to the model's input data folder",
        required=True,
    )

    parser.add_argument(
        "--tv_label",
        dest="train_val_path",
        type=str,
        default="/Data",
        help="Path to the .csv file of train, val labels.",
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
        dest="lr",
        type=float,
        default=0.001,
        help="setting learning rate of optimizer",
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
        type=bool,
        default=True,
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
        type=bool,
        default=False,
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
    args = parser.parse_args()
    return args


# python scripts/train_inception_v3.py --batch=8 --epoch=2 --imgnetweights=True --data=../Data --data_frac=0.001 --tv_label=../Data/Train_val_labels.csv --result=../Results --exp==Default
def main():
    args = parse_arguments()
    tf.config.run_functions_eagerly(True)

    # Parameters
    num_classes = 8
    tag = "inception_v3"

    mlflow.set_experiment(args.experiment)
    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", tag)

    # Load data
    # create compose for both train and validation sets
    compose = Compose(
        transforms=[
            RemovePadding(),
            BenGraham(args.bengraham_scale),
            Resize((args.shape, args.shape), args.keepAspectRatio),
            KerasPreprocess(model_name="inception"),
            # RandomShift(0.2, 0.3),
            # RandomFlipLR(),
            # RandomFlipUD(),
        ]
    )

    # split data
    (df_train, df_val) = split_ODIR(
        path_train_val=args.train_val_path, train_val_frac=args.train_val_fraction
    )
    # create both train and validation sets
    ODIR_dataset = ODIR_Dataset(
        img_folder_path=args.data_folder,
        csv_path=args.train_val_path,
        img_shape=(args.shape, args.shape),
        num_classes=8,
        frac=args.data_fraction,
        transforms=compose,
    )
    train_dataset = ODIR_dataset.subset(df_train)
    val_dataset = ODIR_dataset.subset(df_val)

    # create both train and validation dataloaders
    train_DL = ODIR_Dataloader(dataset=train_dataset, batch_size=args.batch_size)
    val_DL = ODIR_Dataloader(dataset=val_dataset, batch_size=args.batch_size)

    # Model
    model = InceptionV3(num_classes=num_classes)
    if args.imagenet_weights:
        model.load_imagenet_weights()
    # need to check
    if args.weights_path is not None:
        model.load_weights(path=args.weights_path)

    # model trainable and non-trainable parameters
    trainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.model.trainable_weights]
    )
    nonTrainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.model.non_trainable_weights]
    )
    totalParams = trainableParams + nonTrainableParams

    add_args_to_mlflow(args)
    mlflow.log_param("Training data size", len(train_dataset))
    mlflow.log_param("Validation data size", len(val_dataset))
    mlflow.log_param("Total params", totalParams)
    mlflow.log_param("Trainable params", trainableParams)
    mlflow.log_param("Non-trainable params", nonTrainableParams)

    # Optimizer
    # TODO: Define multiple optimizer
    sgd = SGD(
        lr=args.lr, decay=args.decay, momentum=args.momentum, nesterov=args.nesterov
    )

    # Metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        specificity,
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
        monitor=args.early_stopping_monitor,
        patience=args.patience,
        mode=args.early_stopping_mode,
        verbose=args.early_stopping_verbose,
    )

    # Train
    history = model.train(
        epochs=args.epochs,
        loss=args.loss,
        metrics=metrics,
        callbacks=[MlflowCallback(), earlyStoppingCallback],
        optimizer=sgd,
        train_data_loader=train_DL,
        validation_data_loader=val_DL,
        batch_size=args.batch_size,
        steps_per_epoch=len(train_DL),
        shuffle=True,
    )
    print("model trained successfuly.")

    # Save
    print("Saving models weights...")
    model_file = os.path.join(args.result, f"model_weights_{tag}.h5")
    model.save(model_file)
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
