- [Installation](#Installation)
- [Usage](#Usage)
  - [predict.py](#predict.py)
  - [plot_explanation.py](#plot_explanation.py)
  - [train.py](#train.py)

---
## Installation

We suggest you install `Xlnc1DCNN` by using an anaconda environment for installing the required packages and their dependencies easily. *(tested with CentOS 7 and Window 10*)

**Steps**
1. Create an environment using Anaconda.
```
conda create -n Xlnc1DCNN python=3.7.5 --y
conda activate Xlnc1DCNN
```
2. Install Tensorflow. 
- If your machine has a GPU card, we suggest you install Tensorflow through anaconda to let Anaconda install all dependency packages. Then you can run each module on your GPU card.
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
                           [-b {background_10,background_20,background_50,background_100,background_200,background_350,background_500}]
                           [-ub USER_BACKGROUND] [-dpi DPI]
                           [-f [{True,False}]] [--min_len MIN_LEN]
                           [--max_len MAX_LEN]

Plot explanation results from an input FASTA file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input FASTA file (default: None)
  -o OUTPUT, --output OUTPUT
                        output directory (default: output)
  -m MODEL, --model MODEL
                        model file (default: None)
  -b {background_10,background_20,background_50,background_100,background_200,background_350,background_500}, --background {background_10,background_20,background_50,background_100,background_200,background_350,background_500}
                        background distribution to plot explanation results
                        (default: background_350)
  -ub USER_BACKGROUND, --user_background USER_BACKGROUND
                        path to user's background distribution to plot
                        explanation results (default: None)
  -dpi DPI, --dpi DPI   dpi of output images (default: 250)
  -f [{True,False}], --force [{True,False}]
                        force to plot when the input sequences exceed the
                        maximum length. (default: False)
  --min_len MIN_LEN     the minimum of input sequences length to plot
                        (default: 200)
  --max_len MAX_LEN     the maximum of input sequences length to plot
                        (default: 3000)
```
**Example**
```
python plot_explanation.py  -i dataset/example/sample.fasta
```
If your machine can't run the above command, we suggest reducing the background samples size.

```
python plot_explanation.py  -i dataset/example/sample.fasta -b background_10
```

### train.py

`train.py` is the command which allows users to train a new classifier model from a coding and noncoding transcript file. Users can set details of learning by setting optional arguments.

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
                        output directory. (default: model)
  -e EPOCHS, --epochs EPOCHS
                        epochs. (default: 120)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size. (default: 128)
  -m MOMENTUM, --momentum MOMENTUM
                        momentum. (default: 0.9)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate. (default: 0.01)
  --min_len MIN_LEN     the minimum of input sequences length to be trained.
                        (default: 200)
  --max_len MAX_LEN     the maximum of input sequences length to be trained.
                        (default: 3000)
```

**Example**

Train a new classifier model by using default settings.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
```
Train a new classifier model with 10 epochs, 100 batch sizes, and 0.001 learning rate.
```
python train.py dataset/human/training_set/pct_train.fa dataset/human/training_set/lncrna_train.fa 
-e 10 -bs 100 -lr 0.001
```