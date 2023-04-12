"""Author: Javier Abad Martinez"""

import argparse
import random

import numpy as np
import torch
import tqdm
from folktables import ACSDataSource, ACSIncome
from keras.datasets import cifar10, mnist
from sklearn.datasets import make_classification

from acp.methods import ACP_D, ACP_O
from acp.models import AE, ConvolutionalNeuralNetwork, LogisticRegression, NeuralNetwork
from acp.others import (
    APS,
    RAPS,
    CV_plus,
    JK_plus,
    deleted_full_CP,
    ordinary_full_CP,
    split_cp,
)
from acp.utils import evaluate_cp_method, normalize_flatten, scalar, train_test_split

# pylint: disable=no-member
# no error

device = torch.device("cuda")


def experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "function",
        help="CP function to run (full_CP, ACP_D, ordinary_full_CP, ACP_O, SCP, RAPS, APS, CV_plus, JK_plus)",
    )
    parser.add_argument("dataset", help="dataset (synthetic, MNIST, US_Census, CIFAR-10)")
    parser.add_argument("model", help="Neural Network A, B, C, LR or CNN")
    parser.add_argument("--reg", help="value l2 regularization term", type=float, default=0.00001)
    parser.add_argument("--seed", help="initial seed", type=int, default=1000)
    parser.add_argument("--test", help="test set size", type=int, default=100)
    parser.add_argument("--dir", help="output dir", default="results/")
    parser.add_argument(
        "--embedding_size",
        help="embedding size for the autoencoder",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--validation_split",
        help="split for calibration set in SCP",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--epsilon",
        help="value of epsilon for RAPS, APS, JK+ or CV+",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    logs = args.dir
    n_test = args.test
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Loading data
    if args.dataset == "synthetic":
        Ns = np.logspace(1, 5, 13, dtype="int").tolist()[:10]
        n_classes = 5
        P = 10
        X, Y = make_classification(
            max(Ns) + n_test,
            P,
            n_classes=n_classes,
            n_clusters_per_class=1,
            n_informative=3,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=n_test, random_state=args.seed
        )

    elif args.dataset in ["MNIST", "CIFAR-10"]:
        if args.dataset == "MNIST":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load MNIST
        else:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Load CIFAR-10

        # Preprocessing data (2 scenarios: CNN and not-CNN (i.e. NN and LR))
        if args.model == "CNN":
            x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[1], x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[1], x_test.shape[2])
        else:
            x_train = normalize_flatten(x_train)
            x_test = normalize_flatten(x_test)
            autoencoder = AE(
                input_shape=x_train.shape[1],
                embedding_size=args.embedding_size,
                seed=args.seed,
            ).to(device)
            autoencoder.load_state_dict(
                torch.load("models/AE_" + args.dataset + "_" + str(args.embedding_size))
            )
            x_train = np.array(autoencoder.embed(x_train))
            x_test = np.array(autoencoder.embed(x_test))
        y_train = scalar(y_train)
        y_test = scalar(y_test)
        x_test, y_test = x_test[:n_test], y_test[:n_test]
        Ns = [x_train.shape[0]]

    elif args.dataset == "US_Census":
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=["NY"], download=True)
        features, label, group = ACSIncome.df_to_numpy(
            acs_data
        )  # Load income data from US Census New York
        x_train, x_test, y_train, y_test, _, _ = train_test_split(
            features, label, group, test_size=0.1, random_state=args.seed
        )
        x_test, y_test = x_test[:n_test], y_test[:n_test]
        y_train = np.array([1 if x else 0 for x in y_train])
        y_test = np.array([1 if x else 0 for x in y_test])
        Ns = [x_train.shape[0]]
    else:
        print("Dataset not included")
        raise NotImplementedError

    LABELS = np.unique(y_train).tolist()
    out_size = len(LABELS)
    _, P = x_train.shape

    # Folder+file for saving results
    logfile = (
        logs
        + args.function
        + "_"
        + args.dataset
        + "_"
        + str(args.embedding_size)
        + "_"
        + str(args.model)
        + "_"
        + str(args.seed)
    )

    for N in tqdm.tqdm(Ns):
        # Invoking model at each iteration
        if args.model == "A":
            model = NeuralNetwork(
                input_size=P,
                num_neurons=[20, 10],
                out_size=out_size,
                l2_reg=args.reg,
                seed=args.seed,
            )
        elif args.model == "B":
            model = NeuralNetwork(
                input_size=P,
                num_neurons=[100],
                out_size=out_size,
                l2_reg=args.reg,
                seed=args.seed,
            )
        elif args.model == "C":
            model = NeuralNetwork(
                input_size=P,
                num_neurons=[100, 50, 20],
                out_size=out_size,
                l2_reg=args.reg,
                seed=args.seed,
            )
        elif args.model == "LR":
            model = LogisticRegression(
                input_size=P, out_size=out_size, l2_reg=args.reg, seed=args.seed
            )
        elif args.model == "CNN":
            model = ConvolutionalNeuralNetwork(
                channels=x_train.shape[1],
                out_size=out_size,
                l2_reg=args.reg,
                seed=args.seed,
            )
        else:
            print("Model not implemented")
            raise NotImplementedError

        model = model.to(device)

        # Selecting and executing appropriate conformal predictor
        if args.function == "full_CP":
            _, _ = evaluate_cp_method(
                model=model,
                cp_method=deleted_full_CP,
                x_train=x_train[:N, :],
                y_train=y_train[:N],
                x_test=x_test,
                labels=LABELS,
                out_file=logfile,
                seed=args.seed,
            )

        elif args.function == "ordinary_full_CP":
            _, _ = evaluate_cp_method(
                model=model,
                cp_method=ordinary_full_CP,
                x_train=x_train[:N, :],
                y_train=y_train[:N],
                x_test=x_test,
                labels=LABELS,
                out_file=logfile,
                seed=args.seed,
            )

        elif args.function == "ACP_D":
            _, _ = ACP_D(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                labels=LABELS,
                out_file=logfile,
                seed=args.seed,
            )

        elif args.function == "ACP_O":
            _, _ = ACP_O(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                labels=LABELS,
                out_file=logfile,
                seed=args.seed,
            )

        elif args.function == "SCP":
            _, _ = split_cp(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                labels=LABELS,
                out_file=logfile,
                validation_split=args.validation_split,
            )

        elif args.function == "RAPS":
            RAPS(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                y_test,
                labels=LABELS,
                out_file=logfile,
                eps=args.epsilon,
                seed=args.seed,
            )

        elif args.function == "APS":
            APS(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                y_test,
                labels=LABELS,
                out_file=logfile,
                eps=args.epsilon,
                seed=args.seed,
            )

        elif args.function == "CV_plus":
            CV_plus(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                y_test,
                labels=LABELS,
                out_file=logfile,
                eps=args.epsilon,
                seed=args.seed,
            )

        elif args.function == "JK_plus":
            JK_plus(
                model,
                x_train[:N, :],
                y_train[:N],
                x_test,
                y_test,
                labels=LABELS,
                out_file=logfile,
                eps=args.epsilon,
                seed=args.seed,
            )

        else:
            print("CP function not implemented")
            raise NotImplementedError


if __name__ == "__main__":
    experiments()
