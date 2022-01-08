import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


from preprocess import seq_to_array
from tensorflow.compat.v1.keras.backend import get_session
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser(
        description="Predict sequences from a input FASTA file"
    )
    parser.add_argument("-i", "--input", help="Input FASTA file", required=True)
    parser.add_argument("-o", "--output", help="Output directory", default="output")
    parser.add_argument("-m", "--model", help="Model file")
    parser.add_argument(
        "-f",
        "--force",
        help="Force to predict when the input sequences"
        " exceed the maximum length; otherwise, the model"
        "will generate the remaining file.",
        # action="store_true",
        nargs="?",
        type=bool,
        default=False,
        const=False,
        choices=[True, False],
    )
    args = parser.parse_args()

    predict(args)
