- [Installation](#Installation)
- [Usage](#Usage)
  - [predict.py](#predict.py)
  - [plot_explanation.py](#plot_explanation.py)

---
## Installation

## Usage

### predict.py

`predict.py` is the command to predict a input FASTA file.


```
usage: predict.py [-h] -i INPUT [-o OUTPUT] [-m MODEL] [-f [{True,False}]]
                  [--min_len MIN_LEN] [--max_len MAX_LEN]

Predict sequences from a input FASTA file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input FASTA file
  -o OUTPUT, --output OUTPUT
                        Output directory
  -m MODEL, --model MODEL
                        Model file
  -f [{True,False}], --force [{True,False}]
                        Force to predict when the input sequences exceed the
                        maximum length; otherwise, the model will generate the
                        remaining file.
  --min_len MIN_LEN     Minimum of intput sequences length to predict
  --max_len MAX_LEN     Maximum of input sequences length to predict
```

**Example**

```
python predict.py -i dataset/example/sample.fasta
```

### plot_explanation.py 

`plot_explanation.py ` is the command for ploting explanation results of 
the input sequences from the model.
```
usage: plot_explanation.py [-h] -i INPUT [-o OUTPUT] [-m MODEL]
                           [-b {background_50,background_100,background_200,background_350,background_500}]
                           [-ub USER_BACKGROUND] [-dpi DPI]
                           [-f [{True,False}]] [--min_len MIN_LEN]
                           [--max_len MAX_LEN]

Plot explanation results from a input FASTA file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input FASTA file (default: None)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: output)
  -m MODEL, --model MODEL
                        Model file (default: None)
  -b {background_50,background_100,background_200,background_350,background_500}, --background {background_50,background_100,background_200,background_350,background_500}
                        Background distribution to plot explanation results
                        (default: background_350)
  -ub USER_BACKGROUND, --user_background USER_BACKGROUND
                        Path to user's background distribution to plot
                        explanation results (default: None)
  -dpi DPI, --dpi DPI   DPI of output images (default: 250)
  -f [{True,False}], --force [{True,False}]
                        Force to plot when the input sequences exceed the
                        maximum length; otherwise, the modelwill generate the
                        remaining file. (default: False)
  --min_len MIN_LEN     the minimum of intput sequences length to predict
                        (default: 200)
  --max_len MAX_LEN     the maximum of input sequences length to predict
                        (default: 3000)
```
**Example**
```
python plot_explanation.py  -i dataset/example/sample.fasta
```

### train.py

`train.py` is the command to allow user for training a new model from a coding and noncoding transcrtip file. The user can set the details of learning by setting optianal arguments.

```
usage: train.py [-h] [-o OUTPUT] [-e EPOCHS] [-bs BATCH_SIZE] [-m MOMENTUM]
                [-lr LEARNING_RATE] [--min_len MIN_LEN] [--max_len MAX_LEN]
                coding_file noncoding_file

Train model from a input FASTA file

positional arguments:
  coding_file           input FASTA file of coding transcripts.
  noncoding_file        input FASTA file of noncoding transcripts.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory.
  -e EPOCHS, --epochs EPOCHS
                        epochs.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size.
  -m MOMENTUM, --momentum MOMENTUM
                        momentum.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate.
  --min_len MIN_LEN     the minimum of input sequences length to be trained.
  --max_len MAX_LEN     the maximum of input sequences length to be trained.
```

**Example**

Train a new classifer model without any settings.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
```
Train a new classifer model with 10 ephocs, 100 batch sizes and 0.001 learning rate.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
-e 10 -bs 100 -lr 0.001
```