"""Author: Javier Abad Martinez"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats.mstats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

# UTILS FOR STORING RESULTS


def log_to_file(fname, data):
    """Logs a dict 'data' to a file in json lines format."""
    file = open(fname, mode="a", encoding="UTF-8")
    file.write(json.dumps(data) + "\n")
    file.close()


def evaluate_cp_method(model, cp_method, x_train, y_train, x_test, labels, out_file=None, seed=42):
    pvals = []
    prediction_times = []
    for xtest in x_test:
        pvals_xtest = {}
        scores = {}
        start = time.perf_counter()
        for yhat in labels:
            pvals_xtest[yhat], scores[yhat] = cp_method(
                model, x_train, y_train, xtest, yhat, labels=labels, seed=seed
            )
        prediction_times.append(time.perf_counter() - start)
        pvals.append(pvals_xtest)
        if out_file:
            log_to_file(
                out_file,
                {
                    "N": len(x_train),
                    "prediction-times": prediction_times[-1],
                    "p-values": pvals_xtest,
                },
            )

    return pvals, prediction_times


# UTILS FOR ESTIMATING THE GOODNESS OF OUR APPROXIMATION (DISTANCES)


def sigma(x):
    return 1 / (1 + np.exp(-x))


def line(x):
    return 1 - x


def compute_distance(X, Y, metric="euclidean"):
    X = np.expand_dims(np.array(X), axis=0)
    Y = np.expand_dims(np.array(Y), axis=0)
    return pairwise_distances(np.concatenate((X, Y), axis=0), metric=metric)[1][0]


def kendall_tau_distance(X, Y):
    Z1 = np.array([np.array(X["0"]), np.array(X["1"])]).argsort()
    Z2 = np.array([np.array(Y["0"]), np.array(Y["1"])]).argsort()
    n = len(Z1)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(Z1)
    b = np.argsort(Z2)
    ndisordered = np.logical_or(
        np.logical_and(a[i] < a[j], b[i] > b[j]),
        np.logical_and(a[i] > a[j], b[i] < b[j]),
    ).sum()

    return ndisordered / (n * (n - 1))


def spearman_distance(X, Y):
    Z1 = np.array([np.array(X["0"]), np.array(X["1"])])
    Z2 = np.array([np.array(Y["0"]), np.array(Y["1"])])
    dist, _ = spearmanr(Z1, Z2, axis=0)
    return dist


# UTILS FOR (PRE)PROCESSING DATA


def scalar(data):
    res = np.zeros(len(data)).astype(int)
    for i in range(len(data)):
        res[i] = data[i].item()
    return res


def create_set(x, y):
    # Creates training/test set before generator
    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], y[i]])
    return train_data


def check_tensor(X):
    # Makes sure we are using tensors
    if torch.is_tensor(X):
        return X
    if isinstance(X, pd.DataFrame):
        X = X.values
    return torch.tensor(X, requires_grad=False).float()


def make_val_split(X, y, val_split_prop, seed):
    if val_split_prop == 0:
        # return original data
        return X, y, X, y
    else:
        # make actual split
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return X_t, y_t, X_val, y_val


def normalize_flatten(data):
    return data.reshape(data.shape[0], -1) / 255


def flatten_gradient(x):
    flattened_gradient = x[0].flatten()
    for j in range(1, len(x)):
        flattened_gradient = torch.concat((flattened_gradient, x[j].flatten()), dim=0)
    return flattened_gradient


def l2_regularization(model, l2_penalty):
    num_layers = len(model)
    # Only penalizes weights, not bias
    if len(model) == 0:
        return 0
    l2_reg = None
    for i in range(num_layers):
        if l2_reg is None:
            l2_reg = model[i].weight.norm(2).pow(2)
        else:
            l2_reg = l2_reg + model[i].weight.norm(2).pow(2)
    return torch.mul(l2_reg, check_tensor(l2_penalty))  # type: ignore


# UTILS FOR PLOTTING

plt.rc("font", family="Times New Roman")


def print_pvalues(
    results_standard,
    results_optimized,
    xlim,
    ylim,
    label_name,
    label="0",
    save_dir=None,
    size=8,
):
    diff = []
    for _, scores_s, scores_o in zip(
        results_standard.N, results_standard["p-values"], results_optimized["p-values"]
    ):
        diff.append(abs(np.array(scores_s[label]) - np.array(scores_o[label])))
    plt.figure(figsize=(size, size))
    sns.lineplot(x=results_standard.N, y=diff, label=label_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("Training set size")
    plt.ylabel("|Full CP p-value $-$ CP-IF p-value|")
    plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        print("saved!")


def print_nonconformities(
    results_standard,
    results_optimized,
    xlim,
    ylim,
    label_name,
    label="0",
    save_dir=None,
    size=8,
):
    distances = []
    for _, scores_s, scores_o in zip(
        results_standard.N, results_standard["scores"], results_optimized["scores"]
    ):
        distances.append(
            compute_distance(scores_s[label], scores_o[label], metric="l1") / len(scores_o[label])
        )
    plt.figure(figsize=(size, size))
    sns.lineplot(x=results_standard.N, y=distances, label=label_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("|Training set size|")
    plt.ylabel("|Full CP non-conf. scores $-$ CP-IF non-conf. scores|")
    plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        print("saved!")


def print_kendall(
    results_standard,
    results_optimized,
    xlim,
    ylim,
    label_name,
    save_dir=None,
    size=8,
):
    tau_distances = []
    for _, pvalues_standard, pvalues_optimized in zip(
        results_standard.N, results_standard["p-values"], results_optimized["p-values"]
    ):
        tau_distances.append(kendall_tau_distance(pvalues_optimized, pvalues_standard))
    plt.figure(figsize=(size, size))
    sns.lineplot(x=results_standard.N, y=tau_distances, label=label_name)
    plt.xscale("log")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("Training set size")
    plt.ylabel("Kendall's Tau distance")
    plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        print("saved!")


def print_parameters(
    results_standard,
    results_optimized,
    xlim,
    ylim,
    label_name,
    label="0",
    save_dir=None,
    size=8,
):
    distances = []
    for _, all_params_s, all_params_o in zip(
        results_standard.N, results_standard["params"], results_optimized["params"]
    ):
        aux_dist = 0
        for params_s, params_o in zip(all_params_s[label], all_params_o[label]):
            aux_dist += compute_distance(params_s, params_o, metric="l1") / len(params_o)
        distances.append(aux_dist / len(all_params_s[label]))
    plt.figure(figsize=(size, size))
    sns.lineplot(x=results_standard.N, y=distances, label=label_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("Training set size")
    plt.ylabel("|Full CP params. $-$ CP-IF params.|")
    plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        print("saved!")


def plot_accum_error(epsilons, data, y_test, file):
    _, ax = plt.subplots()
    for epsilon in epsilons:
        accum_error = compute_accum_error(epsilon, data, y_test)
        plt.step(
            np.arange(1, len(y_test) + 1),
            accum_error,
            label=r"$\epsilon$ = {}".format(epsilon),
            linewidth=3,
        )
    ax.legend(fontsize="20", shadow=True, facecolor="white", framealpha=1)
    plt.xlim([0, len(y_test)])
    plt.xticks(fontsize="18")
    plt.yticks([1, 5, 10, 15, 20], fontsize="18")
    plt.ylim([0, 20])
    plt.ylabel("Accumulative error", fontsize="22")
    plt.xlabel("n", fontsize="22")
    plt.grid()
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, dpi=500)


# UTILS TO CALCULATE EFFICIENCY METRICS


def fuzziness(data, y_test):
    fuzziness_optimized = {}
    for N in np.unique(data.N):
        fuzziness_optimized[N] = []
        for _, pval in zip(y_test, data[data.N == N]["p-values"]):
            pval = dict(pval).values()
            fuzziness_optimized[N].append(sum(pval) - max(pval))

    return fuzziness_optimized


def S_criterion(data, y_test):
    average_sums = {}
    for N in np.unique(data.N):
        average_sums[N] = []
        for _, pval in zip(y_test, data[data.N == N]["p-values"]):
            pval = dict(pval).values()
            average_sums[N].append(sum(pval))
    return average_sums


def U_criterion(data, y_test):
    unconfidence = {}
    for N in np.unique(data.N):
        unconfidence[N] = []
        for _, pval in zip(y_test, data[data.N == N]["p-values"]):
            pval = list(dict(pval).values())
            pval_copy = pval.copy()
            pval_copy.remove(max(pval_copy))
            unconfidence[N].append(max(pval_copy))
    return unconfidence


def OU_criterion(data, y_test):
    unconfidence = {}
    for N in np.unique(data.N):
        unconfidence[N] = []
        for y, pval in zip(y_test, data[data.N == N]["p-values"]):
            pval = list(dict(pval).values())
            pval_copy = pval.copy()
            pval_copy.pop(int(y))
            unconfidence[N].append(max(pval_copy))
    return unconfidence


def OF_criterion(data, y_test):
    sums = {}
    for N in np.unique(data.N):
        sums[N] = []
        for y, pval in zip(y_test, data[data.N == N]["p-values"]):
            pval = list(dict(pval).values())
            pval_copy = pval.copy()
            pval_copy.pop(int(y))
            sums[N].append(sum(pval_copy))
    return sums


def epsilon_curves(data, epsilons):
    intervals = {}
    for epsilon in epsilons:
        for j in range(len(data["p-values"])):
            labels = []
            for label, pvalue in data["p-values"][j].items():
                if pvalue > epsilon:
                    labels.append(label)
            if epsilon not in intervals:
                intervals[epsilon] = [len(labels)]
            else:
                intervals[epsilon].append(len(labels))
    return [np.mean(x) for x in intervals.values()]


def find_min_epsilon(data, y_test, sample, epsilons):
    reversed_epsilons = reversed(epsilons)
    for epsilon in reversed_epsilons:
        labels = []
        for label, pvalue in data["p-values"][sample].items():
            if pvalue > epsilon:
                labels.append(label)
        if str(y_test[sample]) in labels:
            return epsilon
        if type(y_test[sample]) == np.ndarray:
            if str(y_test[sample][0]) in labels:
                return epsilon
    return epsilons[0]


def get_average_interval(data, sample, epsilons):
    intervals = {}
    for epsilon in epsilons:
        labels = []
        for label, pvalue in data["p-values"][sample].items():
            if pvalue > epsilon:
                labels.append(label)
        if epsilon not in intervals:
            intervals[epsilon] = [len(labels)]
        else:
            intervals[epsilon].append(len(labels))
    return [x[0] for _, x in intervals.items()]


def get_global_coverage(data, y_test, epsilons):
    coverage = {}
    for epsilon in epsilons:
        cov = 0
        for i in range(len(data["p-values"])):
            labels = []
            for label, pvalue in data["p-values"][i].items():
                if pvalue > epsilon:
                    labels.append(label)
            if str(y_test[i]) in labels:
                cov = cov + 1
            if type(y_test[i]) == np.ndarray:
                if str(y_test[i][0]) in labels:
                    cov = cov + 1
        coverage[epsilon] = cov / len(y_test)
    return [x for _, x in coverage.items()]


def compute_accum_error(epsilon, data, y_test):
    accum_error = np.zeros(len(data) + 1)
    for i in range(len(data["p-values"])):
        labels = []
        for label, pvalue in data["p-values"][i].items():
            if pvalue > epsilon:
                labels.append(label)
        if type(y_test[i]) == np.ndarray:
            if str(y_test[i][0]) not in labels:
                accum_error[i + 1] = accum_error[i] + 1
            else:
                accum_error[i + 1] = accum_error[i]
        else:
            if str(y_test[i]) not in labels:
                accum_error[i + 1] = accum_error[i] + 1
            else:
                accum_error[i + 1] = accum_error[i]
    return accum_error[1:]


def compute_accum_acc(epsilon, data, y_test):
    accum_acc = np.zeros(len(data) + 1)
    for i in range(len(data["p-values"])):
        labels = []
        for label, pvalue in data["p-values"][i].items():
            if pvalue > epsilon:
                labels.append(label)
        if type(y_test[i]) == np.ndarray:
            if str(y_test[i][0]) in labels:
                accum_acc[i + 1] = accum_acc[i] + 1
            else:
                accum_acc[i + 1] = accum_acc[i]
        else:
            if str(y_test[i]) in labels:
                accum_acc[i + 1] = accum_acc[i] + 1
            else:
                accum_acc[i + 1] = accum_acc[i]
    return accum_acc[1:]


def MSE(y, y_hat):
    return np.linalg.norm(y - y_hat)
