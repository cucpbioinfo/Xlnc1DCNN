import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import seq_to_array
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import get_session


def predict(args, threshold=0.5):
    """Predict sequences from a input FASTA file"""

    directory = args.output
    input_file = args.input
    min_len = args.min_len
    max_len = args.max_len

    if args.force:
        filter_seq = False
    else:
        filter_seq = True

    if not os.path.exists(directory):
        os.makedirs(directory)

    seq, seq_names, _ = seq_to_array(
        input_file,
        outdir=directory,
        filter_seq=filter_seq,
        min_len=min_len,
        max_len=max_len,
    )
    print("Loading Model")
    if args.model:
        print("    Use user's model")
        model = load_model(f"{args.model}")
    else:
        print("    Use pretrained model")
        model = load_model("model/model.hdf5")
    print("    Done")
    print("Predicting")
    prediction = model.predict(seq)[:, 1]
    print("    Done")

    # Output file
    output_file = pd.DataFrame(seq_names, columns=["name"])
    output_file["probability"] = prediction
    output_file["prediction"] = np.where(prediction > threshold, "lncRNA", "mRNA")
    output_file.to_csv(f"{directory}/prediction.csv")


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
    parser.add_argument(
        "--min_len",
        help="Minimum of intput sequences length to predict",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--max_len",
        help="Maximum of input sequences length to predict",
        default=3000,
        type=int,
    )

    args = parser.parse_args()

    predict(args)
