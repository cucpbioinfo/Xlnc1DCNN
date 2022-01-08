import re
import numpy as np
from Bio import SeqIO
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


MINIMUM_LENGTH = 200  # Minimum sequence length to determine as lncRNA
MAXIMUM_LENGTH = 3000  # Maximum length for the CNN model
KMER_SIZE = 5  # Set K-mer Size
KMER_WORDS = "atcgn"  # Set K-mer words
MODE_OHE = True


def string_to_array(string):
    """Return lower case string and convert to array.
    """
    string = string.lower()
    string = re.sub("[^acgt]", "z", string)
    string_array = np.array(list(string))
    return string_array


def one_hot_encoder(array, encode_seq=["a", "c", "g", "t", "z"], mode=MODE_OHE):
    """Return array of one hot encode of input array.
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(encode_seq))

    integer_encoded = label_encoder.transform(array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    if mode is not None:
        onehot_encoded[np.where(onehot_encoded[:, 4] == 1), :] = 1

    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


def seq_to_array(
    fa_file, min_len=MINIMUM_LENGTH, max_len=MAXIMUM_LENGTH, filter_seq=True, outdir=""
):
    """Import raw file of sequences then Return to one hot encode format.

    Parameters
    ----------
    fa_file : string
        Path of input file.

    min_len : int, Default 200 (Select only long non coding RNA)
        The minimum length of raw file to be encoded.

    max_len : int, Default 2500 (Maximum length of selected sequence)
        The maximum length of raw file to be encoded.

    filter_seq_seq : bool, optional
        Filter sequences that their length exceeds max_len or below min_len.
        If set filter_seq to False, this function will trim the remaining sequences
        that exceed max_len to the length of max_len instead of discard them.
        , by default True

    outdir : str, optional
        Output directory of the remaining sequences, if some sequences got trim.

    Returns
    -------
    Return array of raw file that get one hot encoded.
    """
    if isinstance(fa_file, str):
        obj_iter = SeqIO.parse(fa_file, "fasta")
        print("Importing", str(fa_file))
        print("    One Hot Encoding", str(fa_file))
    else:
        raise TypeError("fa_file must be str")

    print("    Settings >> Min seq len:", min_len, "Max seq len:", max_len)

    if not filter_seq:
        "     >> Disable filter_seqing sequences"

    encode_seq = []  # Initialize list for store encoding sequences

    count = 0  # Count the number of seq from fa
    low_count = 0  # Count the number of seq below min_loen
    high_count = 0  # Count the number of seq above max_len

    seq_names = []
    all_seq_len = []
    rem_seq = []

    for seq_record in obj_iter:
        seq_len = len(seq_record.seq)

        # Seq len < Min
        if seq_len < min_len:
            low_count += 1
            rem_seq.append(seq_record)
            continue

        # Seq len > Max
        elif seq_len > max_len:
            high_count += 1
            if not filter_seq:
                rem_seq.append(seq_record)
                continue

        # Encode the sequence with one hot encode
        ohe_tmp = one_hot_encoder(string_to_array(str(seq_record.seq)))

        # The length of sequnce that have to fill with 0
        # Pad til seq length is eqaul to max_len
        if seq_len < max_len:
            pad_len = max_len - seq_len

            pad_seq = np.pad(
                ohe_tmp, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )
        else:
            pad_seq = ohe_tmp[:max_len]

        # Append to stored list
        encode_seq.append(pad_seq)
        seq_names.append(seq_record.name)
        all_seq_len.append(seq_len)
        count += 1

    print("     Sequences Length:", count)
    print("     Selected Length:", len(encode_seq))

    if low_count > 0 or (high_count > 0 and not filter_seq):
        with open(f"{outdir}/remaining.fasta", "w") as output_handle:
            SeqIO.write(rem_seq, output_handle, "fasta")

    # Print the result
    print(" Below(", min_len, "):", low_count, ", Exceed(", max_len, "):", high_count)

    encode_seq = np.array(encode_seq)
    print("Done\n")
    return encode_seq, seq_names, all_seq_len


def get_train_data(
    lncRNA_path, pcts_path, min_len=MINIMUM_LENGTH, max_len=MAXIMUM_LENGTH
):
    """Return both data and its label that ready to train in Keras
    """

    # Encode raw data by using one hot encode
    x_val_lncRNA, _, _ = seq_to_array(
        lncRNA_path, min_len=min_len, max_len=max_len
    )  # Encode lncRNA
    x_val_pcts, _, _ = seq_to_array(
        pcts_path, min_len=min_len, max_len=max_len
    )  # Encode Pcts

    # Combine lncRNA and Pcts
    x_val = np.vstack((x_val_lncRNA, x_val_pcts))

    # Get labels of lncRNA and Pcts
    y_val = list(np.repeat(1, len(x_val_lncRNA))) + list(np.repeat(0, len(x_val_pcts)))

    # Convert to categorical type for being input format in Keras
    y_val = to_categorical(y_val)

    return x_val, y_val
