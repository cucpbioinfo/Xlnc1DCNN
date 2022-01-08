import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import get_train_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dropout,
    Dense,
    GlobalAveragePooling1D,
    BatchNormalization,
    Activation,
    MaxPooling1D,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import backend as Kd


def create_model(
    kernel_size, stride_size, filter_size, max_len, epochs=120, lrate=0.01, momentum=0.9
):
    model = Sequential()
    model.add(
        Conv1D(
            filter_size,
            kernel_size,
            strides=stride_size,
            input_shape=(max_len, 4),
            padding="same",
            activation="relu",
            kernel_constraint=max_norm(3),
        )
    )

    model.add(MaxPooling1D(pool_size=(2), padding="same"))
    model.add(Dropout(0.3))

    model.add(
        Conv1D(
            filter_size,
            kernel_size,
            strides=stride_size,
            activation="relu",
            padding="same",
            kernel_constraint=max_norm(3),
        )
    )
    model.add(MaxPooling1D(pool_size=(2), padding="same"))
    model.add(Dropout(0.3))

    model.add(
        Conv1D(
            filter_size,
            kernel_size,
            strides=stride_size,
            activation="relu",
            padding="same",
            kernel_constraint=max_norm(3),
        )
    )
    model.add(MaxPooling1D(pool_size=(2), padding="same"))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_constraint=max_norm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_constraint=max_norm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    print(model.summary())
    return model


def train(args):

    # Input settings
    lncrna = args.noncoding_file
    pct = args.coding_file
    directory = args.output

    # Model settings
    batch_size = args.batch_size
    epochs = args.epochs
    lrate = args.learning_rate
    momentum = args.momentum
    min_len = args.min_len
    max_len = args.max_len

    train, labels = get_train_data(lncrna, pct, min_len, max_len)

    X_train, X_val, y_train, y_val = train_test_split(
        train, labels, stratify=labels[:, 1], test_size=0.1, random_state=44
    )

    savepath = f"{directory}/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath = savepath + "user_model.hdf5"
    checkpoint = ModelCheckpoint(
        savepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )

    model = create_model(
        57, 1, 120, max_len=max_len, epochs=epochs, lrate=lrate, momentum=momentum
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser(
        description="Train model from a input FASTA file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("coding_file", help="input FASTA file of coding transcripts.")
    parser.add_argument(
        "noncoding_file", help="input FASTA file of noncoding transcripts."
    )
    parser.add_argument("-o", "--output", help="output directory.", default="model")
    parser.add_argument("-e", "--epochs", help="epochs.", default=120, type=int)
    parser.add_argument(
        "-bs", "--batch_size", help="batch size.", default=128, type=int
    )
    parser.add_argument("-m", "--momentum", help="momentum.", default=0.9, type=float)
    parser.add_argument(
        "-lr", "--learning_rate", help="learning rate.", default=0.01, type=float
    )

    parser.add_argument(
        "--min_len",
        help="the minimum of input sequences length to be trained.",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--max_len",
        help="the maximum of input sequences length to be trained.",
        default=3000,
        type=int,
    )

    args = parser.parse_args()

    train(args)
