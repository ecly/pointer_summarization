# Pointer Summarization

Accompanying code for Master's Thesis written at the IT-University in Copenhagen on *Neural Automatic Summarization*.
Focused around reimplementing, optimizing and improving the Pointer Generator from [See et al. 2017](https://arxiv.org/abs/1704.04368).
For all the experiments described in the thesis, refer to the [experiments](experiments) folder for corresponding configuration files.
For the chunk-based approaches described in the thesis, a new repository will be published in the near future. 
All code has been formatted using [black](https://github.com/python/black).

A pretrained base model can be downloaded [here](https://web.tresorit.com/l#TQNev2-hWI5W81dfh79Z6Q).  
If set up correctly, evaluation on the CNNDM test set should produce the following ROUGE F1-scores:  
```bash 
ROUGE-1: 39.03 
ROUGE-2: 17.01 
ROUGE-L: 36.25
```
A pretrained base model with a vocabulary of size 20K and unknown token blocking can be download [here](https://web.tresorit.com/l#Vk7w4jdkLekIZwJBUQHUcQ).  
Evaluation thereof on the CNNDM test should produce the following ROUGE F1-scores:
```bash 
ROUGE-1: 39.26
ROUGE-2: 17.16
ROUGE-L: 36.42
```

**NOTE**: the code provided supports several experimental features that were not discussed in the thesis.

## Overview:
1. [Quick Start](#quick-start)
2. [Setup](#setup)
    1. [Dependencies](#dependencies)
    2. [Data](#data)
        1. [Newsroom](#newsroom)
        2. [New York Times](#new-york-times)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [License](#license)

# Quick Start
- Install all dependencies according to the [Dependencies](#dependencies) section
- Download preprocessed CNNDM data [here](https://web.tresorit.com/l#Ha8s-v4PCbsyxe9X00Ojnw) and extract it into the [data](data) folder
- Train and evaluate a base model with `python train.py -cfg experiments/base.yaml --eval`
- Alternatively, evaluate a pretrained model with `python evaluate.py log/base.tar`

# Setup
All development and testing was done using PyTorch 1.0.1 and Python 3.7.3, but other versions may work fine.

## Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### ROUGE
To use the official ROUGE 155 Perl implementation, download it [here](https://web.tresorit.com/l#BPSRMOtfRtK3PE8vjL-U9Q) and extract it into the [tools](tools) folder.
You should now have 'ROUGE-1.5.5' folder inside your tools folder.
The python wrapper [pyrouge](https://pypi.org/project/pyrouge/) is set up to use now extracted folder.
Alternatively, modify [evaluate.py](evaluate.py) to use a system-wide ROUGE configuration, or evaluate using [py-rouge](https://pypi.org/project/py-rouge/) (see [Evaluation](#evaluation) section).

The official ROUGE 155 Perl implementation, rely on libraries that may not be installed by default.  
We provide instructions for Arch Linux and Ubuntu:
- On Arch Linux: `sudo pacman -S perl-xml-xpath`
- On Ubuntu: `sudo apt-get install libxml-parser-perl`

For possible tips for installation on Windows, or in general, refer to [this](https://stackoverflow.com/questions/47045436/how-to-install-the-python-package-pyrouge-on-microsoft-windows) StackOverflow post.


## Data
We provide preprocessed data for CNNDM [here](https://web.tresorit.com/l#Ha8s-v4PCbsyxe9X00Ojnw). 
The tarfile includes train, dev and test set, as well as vocabularies both with and without proper noun filtering.
For easy setup, extract the tarfile into the data directory.

To manually preprocess CNNDM, refer to Abigail See's [repository](https://github.com/abisee/cnn-dailymail).  
To download already preprocessed CNNDM data according to Abigail See's repository, refer to Jaffer Wilson's [repository](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).  
Once downloaded/preprocessed, you will have files in binary format. 
We use tsv-files instead, to which binary files can be converted, using the following script:
```bash
# note: 'bin_to_tsv.py' depends on TensorFlow which is not in requirements.txt
python tools/bin_to_tsv.py path/to/train.bin data/cnndm_abisee_train.tsv 
```

To create a new vocabulary file of size 50,000 from a tsv-file, use [vocabulary.py](vocabulary.py):
```bash
python vocabulary.py cnndm_abisee_train.tsv cnndm_abisee.vocab 50000
```

Lastly, in case one wants to train using pretrained GLoVe embeddings, download them from the [official GLoVe website](https://nlp.stanford.edu/projects/glove://nlp.stanford.edu/projects/glove/), and convert them to a compatible format using:
```bash
python tools/glove_to_w2v.py path/to/glove.6B.100d.txt data/glove100.w2v 
```

### Newsroom
The Newsroom dataset can be downloaded from its [official website](https://summari.es/).
For preprocessing, we supply a [simple script](tools/preprocess_newsroom.py) using NLTK, which can be used as follows:
```bash
python tools/preprocess_newsroom.py release/train.jsonl.gz newsroom_train.tsv
```


### New York Times
The New York Times Annotated Corpus can be acquired through [LDC](https://catalog.ldc.upenn.edu/LDC2008T19).
For preprocessing, we follow [Paulus et al. 2017](https://arxiv.org/abs/1705.04304), using a [script](tools/preprocess_nyt.py) supplied by the authors. Note that this requires a local [CoreNLP](https://github.com/stanfordnlp/CoreNLP) server and that the script takes a long time to run.

## Training
To train a model using one of the configuration files supplied, use the following command:
```bash
python train.py -cfg experiments/base.yaml
# config files can also have paramterers overwritten on the fly
python train.py -cfg experiments/base.yaml --rnn_cell lstm
```

To resume a model that was cancelled/interrupted use:
```bash
python train.py --resume_from log/model.tar
# optionally, some parameters can be changed when resuming
python train.py --resume_from log/model.tar --batch_size 32
```

To resume training a model that has trained without coverage, while also converting it to now use coverage, use:
```bash
# note that this is only tested with default attention configuration
python train.py --resume_from log/model.tar --convert_to_coverage
```

See [train.py](train.py) and [config.py](config.py) for all possible options.

## Evaluation
To evaluate a model on some test set using official ROUGE 155 Perl implementation:
```bash
python evaluate.py log/model.tar path/to/test_set.tsv
```

To evaluate using [py-rouge](https://pypi.org/project/py-rouge/), a Python reimplementation with less dependencies, use:
```bash
# note that py-rouge does not produce rouge scores identical to the perl implementation
python evaluate.py log/model.tar path/to/test_set.tsv --use_python
```

We support many different test-time parameters that can be given to evaluate.
Refer to [config.py](config.py) and possibly [beam_search.py](beam_search.py) for all options.
Some example uses of said options follow:
```bash
python evaluate.py log/model.tar path/to/test_set.tsv --length_normalize wu --length_normalize_alpha 1.0
python evaluate.py log/model.tar path/to/test_set.tsv --beam_size 8
python evaluate.py log/model.tar path/to/test_set.tsv --block_ngram_repeat 3
python evaluate.py log/model.tar path/to/test_set.tsv --block_unknown

# save summaries and configuration to a json-file
python evaluate.py log/model.tar path/to/test_set.tsv --save
```
Note that [util.py](util.py) can be used to easily inspect the attributes of a model (see module documentation for further information).

## License
**NOTE:** [preprocess_nyt.py](tools/preprocess_nyt.py), [plot.py](tools/plot.py) and [jsonl.py](tools/jsonl.py) all have separate licenses. See each file's header for specifics.  
All other code is distributed under MIT: 
___

MIT License

Copyright (c) 2019 Emil Lynegaard

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
