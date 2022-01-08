import os
import shap
import joblib
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import seq_to_array
from preprocess.plot import plot_shap_amino

from tensorflow.compat.v1.keras.backend import get_session
from tensorflow.keras.models import load_model


def plot(args):
    directory = args.output
    input_file = args.input
    min_len = args.min_len
    max_len = args.max_len
    background_name = args.background
    if args.force:
        filter_seq = False
    else:
        filter_seq = True

    if not os.path.exists(directory):
        os.makedirs(directory)

    seq, seq_names, seq_len = seq_to_array(
        input_file,
        outdir=directory,
        min_len=min_len,
        max_len=max_len,
        filter_seq=filter_seq,
        generate_output=False,
    )

    print("Loading Model")
    if args.model:
        print("    Use user's model")
        model = load_model(f"{args.model}")
    else:
        print("    Use pretrained model")
        model = load_model("model/model.hdf5")
    print("    Done")

    print("Loading Background")

    if args.user_background:
        background = joblib.load(args.user_background)
    else:
        background = joblib.load(f"dataset/background/{background_name}.joblib")
    print("    Done")

    print("Initializa DeepSHAP")
    exp_model = shap.DeepExplainer(model, background)
    print("    Done")

    print("Start Ploting")
    print(
        " If a transcript name has '|' inside,",
        "it will automatically replace by '_' to prevent file name error.",
    )
    count = 1
    sl = len(seq_len)
    for (s, name, length) in zip(seq, seq_names, seq_len):
        print(f"    Progress >> {count}/{sl}", end="\r")

        file_path = f"{directory}/explaination_results/"

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if "|" in name:
            save_path = file_path + f"{name.replace('|', '_')}.png"
        else:
            save_path = file_path + f"{name}.png"
        shap_values = exp_model.shap_values(s.reshape(1, 3000, 4))[1][0]
        plot_shap_amino(
            shap_values,
            length,
            name,
            save_path=save_path,
            trimp_shap=True,
            dpi=args.dpi,
        )

        count += 1
    print()
    print("    Done")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser(
        description="Plot explanation results from a input FASTA file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", help="Input FASTA file", required=True)
    parser.add_argument("-o", "--output", help="Output directory", default="output")
    parser.add_argument("-m", "--model", help="Model file")
    parser.add_argument(
        "-b",
        "--background",
        help="Background distribution to plot explanation results",
        default="background_350",
        choices=[
            "background_50",
            "background_100",
            "background_200",
            "background_350",
            "background_500",
        ],
    )

    parser.add_argument(
        "-ub",
        "--user_background",
        help="Path to user's background distribution to plot explanation results",
    )
    parser.add_argument(
        "-dpi", "--dpi", help="DPI of output images", type=int, default=250
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force to plot when the input sequences exceed the maximum length.",
        # action="store_true",
        nargs="?",
        type=bool,
        default=False,
        const=False,
        choices=[True, False],
    )
    parser.add_argument(
        "--min_len",
        help="the minimum of intput sequences length to plot",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--max_len",
        help="the maximum of input sequences length to plot",
        default=3000,
        type=int,
    )

    args = parser.parse_args()

    plot(args)
