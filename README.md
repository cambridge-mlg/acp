# Approximate full Conformal Prediction

This repository contains the Python implementation of [Approximating Full Conformal Prediction at Scale via Influence Functions](https://arxiv.org/abs/2202.01315).

* [Overview](#overview)
* [Contents](#contents)
* [Third-party software](#third-party-software)
* [Usage](#usage)
* [Tutorial Notebook](#tutorial-notebook)
* [Experiments](#experiments)
* [Reference](#reference)

## Overview

![alt text](resources/ACP.JPG)

Approximate full Conformal Prediction (ACP) outputs a prediction set that contains the true label with at least a probability specified by the practicioner. In large datasets, ACP inherits the statistical power of the highly efficient full Conformal Prediction. The method works as a wrapper for any differentiable ML model.

## Contents

This repository is organized as follows. In the folder `src/acp` you can find the following modules:

 - `methods.py` Python implementation of the ACP algorithms.
 - `others.py` Python implementation of the comparing methods (SCP, APS, RAPS, CV+, JK+).
 - `wrapper.py` Python implementation of ACP as a wrapper for any differentiable PyTorch model. See `models.py` for examples.
 - `models.py` Examples of models compatible with `wrapper.py` (e.g., logistic regression, neural network, convolutional neural network).
 - `experiments.py` Python file to run the experiments from the command line.
 - `models/` Saved models.

The folder  `src/third_party/` contains additional third-party software.
 
## Third-party software

We include the following third-party packages for comparison with ACP:

- [RAPS](https://github.com/aangelopoulos/conformal_classification)
- [APS, CV+, JK+](https://github.com/msesia/arc)
 

## Usage

### Requirements

* python 3.7 or higher
* numpy
* torch
* tqdm
* pandas

For `experiments.py` and `ACP_Tutorial.ipynb`:

* matplotlib
* tensorflow
* keras
* folktables
* scikit-learn
* seaborn
* scipy

### Installation
ACP can be utilized as a fully-independent `pip` package. You can download the framework by running the following command in the terminal:

```bash
pip install approx-cp
```
In order to use ACP in your own models, just include the following imports in your file:

```bash
from acp.wrapper import ACP_D, ACP_O #Deleted scheme (ACP_D) and ordinary scheme (ACP_O)
```
Alternatively, you can clone this repo by running:

```bash
git clone https://github.com/cambridge-mlg/acp
cd acp
```
And install the ACP Python package in a customizable conda environment:

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install --upgrade pip
pip install -e .         
```
Now, just include the import:

```bash
from acp.wrapper import ACP_D, ACP_O
```

### Constructing prediction sets with ACP

ACP works as a wrapper for any PyTorch model with `.fit()` and `.predict()` methods. Once you instantiate your model, you can generate tight prediction sets that contain the true label with a specified probability. Here is an example with synthetic data:

```bash
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from acp.models import NeuralNetwork
from acp.wrapper import ACP_D

X, Y = make_classification(n_samples = 1100, n_features = 10, n_classes = 2, n_clusters_per_class = 1, n_informative = 3, random_state = 42)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 100, random_state = 42)
model = NeuralNetwork(input_size = 10, num_neurons = [20, 10], out_size = 2, seed = 42, l2_reg = 0.01)

ACP = ACP_D(Xtrain, Ytrain, model, seed = 42, verbose = True)
sets = ACP.predict(Xtest, epsilon = 0.1, out_file = "results/test")
```

## Tutorial Notebook

For a tutorial on how to use ACP and how to create the plots in the paper, see the following notebook:

* [Constructing prediction sets with ACP](src/ACP_Tutorial.ipynb)

## Experiments

To easily run experiments with ACP and the comparing methods, use `python3 experiments.py <function> <dataset> <model>`.

The first argument, `<function>`, specifies the CP function. It should be one of the following:
 
* `full_CP`: full Conformal Prediction with the deleted scheme;
* `ordinary_full_CP`: full Conformal Prediction with the ordinary scheme;
* `ACP_D`: ACP with the deleted scheme;
* `ACP_O`: ACP with the ordinary scheme;
* `SCP`: Split Conformal Prediction;
* `RAPS`: Regularized Adaptive Prediction Sets [(Angelopoulos et al.)](https://arxiv.org/abs/2009.14193);
* `APS`:  Adaptive Prediction Sets [(Romano et al.)](https://arxiv.org/abs/2006.02544);
* `CV_plus`:  Cross-validation+ [(Romano et al.)](https://arxiv.org/abs/2006.02544);
* `JK_plus`: Jackknife+ [(Romano et al.)](https://arxiv.org/abs/2006.02544).

The second argument, `<dataset>`, specifies the dataset. It should be one of the following:

* `synthetic`: synthetic data using scikit-learn's make_classification();
* `MNIST`: MNIST dataset;
* `CIFAR-10`: CIFAR-10 dataset; 
* `US_Census`: ACSIncome data for the state of New York from the [US Census dataset](https://github.com/zykls/folktables).

The third argument, `<model>`, specifies the model. It should be one of the following:

* `A`: setting 10-20 in the paper;
* `B`: setting 100;
* `C`: setting 100-50-20; 
* `LR`: setting LR;
* `CNN`: setting CNN.

For all options, see `python3 experiments.py --help`:

```
usage: experiments.py [-h] [--reg REG] [--seed SEED] [--test TEST] [--dir DIR] 
                      [--embedding_size EMBEDDING_SIZE] [--validation_split VALIDATION_SPLIT] 
                      [--epsilon EPSILON] function dataset model

positional arguments:
  function              CP function to run (full_CP, ACP_D, ordinary_full_CP, ACP_O, SCP, RAPS, APS, CV_plus, JK_plus)
  dataset               dataset (synthetic, MNIST, US_Census, CIFAR-10)
  model                 Neural Network A, B, C, LR or CNN

optional arguments:
  -h, --help            show this help message and exit
  --reg REG             value l2 regularization term
  --seed SEED           initial seed
  --test TEST           test set size
  --dir DIR             output dir
  --embedding_size EMBEDDING_SIZE
                        embedding size for the autoencoder
  --validation_split VALIDATION_SPLIT
                        split for calibration set in SCP
  --epsilon EPSILON     value of epsilon for RAPS, APS, JK+ or CV+
```

## Reference

J. Abad Martinez, U. Bhatt, A. Weller and G. Cherubin. Approximating Full Conformal Prediction at Scale via Influence Functions. Association for the Advancement of Artificial Intelligence Conference on Artificial Intelligence (AAAI), 2023.

 BiBTeX:

```
@inproceedings{Abad2023ApproximatingFC,
  title={Approximating Full Conformal Prediction at Scale via Influence Functions},
  author={Javier Abad and Umang Bhatt and Adrian Weller and Giovanni Cherubin},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```



