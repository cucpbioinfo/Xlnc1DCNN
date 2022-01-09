- [Installation](#Installation)
- [Usage](#Usage)
  - [predict.py](#predict.py)
  - [plot_explanation.py](#plot_explanation.py)
  - [train.py](#train.py)

---
## Installation

We suggest you to install this package by using an anaconda environment to install the required packages and their dependencies easily. *(test with CentOS 7 and Window 10*)

**Steps**
1. Create an enviroment using anaconda.
```
conda create -n Xlnc1DCNN python=3.7.5 --y
conda activate Xlnc1DCNN
```
2. Install Tensorflow. 
- If your machine has a GPU card, we suggest you install Tensorflow through anaconda to let anaconda install all dependency packages. Then you can run each module through a GPU card.
```
conda install -c anaconda tensorflow-gpu=2.1.0 --y
```
- Otherwise,

```
pip install tensorflow==2.1.0
```

3. Install the remaining required packages.

```
pip install -r requirement.txt
```

4. Test.
```
python predict.py -i dataset/example/sample.fasta
```


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
                        input FASTA file. (default: None)
  -o OUTPUT, --output OUTPUT
                        output directory. (default: output)
  -m MODEL, --model MODEL
                        path to the model file. (default: None)
  -f [{True,False}], --force [{True,False}]
                        Force to predict when the input sequences exceed the
                        maximum length; otherwise, the model will generate the
                        remaining file. (default: False)
  --min_len MIN_LEN     the minimum of intput sequences length to predict
                        (default: 200)
  --max_len MAX_LEN     the maximum of input sequences length to predict
                        (default: 3000)
```

**Example**

The model will predict only sequences that their length are <= `max_len`. The remaining 
sequences will be generate into a `remaining_<file_name>.fasta` file.
```
python predict.py -i dataset/example/sample.fasta
```

Or, you can force the model to predict all length of sequences by using this command.
```
python predict.py -i dataset/example/sample.fasta -f True
```

### plot_explanation.py 

`plot_explanation.py` is the command for ploting explanation results of 
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
If your machine can't run the above command, we suggest you to reducte the background samples size.

```
python plot_explanation.py  -i dataset/example/sample.fasta -b background_50
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

Train a new classifer model by using default settings.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
```
Train a new classifer model with 10 ephocs, 100 batch sizes and 0.001 learning rate.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
-e 10 -bs 100 -lr 0.001
```