"""Author: Javier Abad Martinez"""

import copy
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from acp.utils import check_tensor, create_set, log_to_file, train_test_split
from third_party.ARC.arc import methods
from third_party.RAPS.RAPS_conformal import ConformalModel

# pylint: disable=no-member
# no error

device = torch.device("cuda")


def deleted_full_CP(model, x_train, y_train, xtest, yhat, labels=None, seed=42):
    """Runs deleted full CP to make a prediction for (xtest, yhat)."""
    N = len(x_train)
    x_tmp = np.row_stack((x_train, [xtest]))
    y_tmp = np.concatenate((y_train, [yhat]))
    alphas = np.zeros(len(x_tmp))

    for i, (x, y) in enumerate(zip(x_tmp, y_tmp)):
        new_model = copy.deepcopy(model)
        new_model = new_model.to(device)

        new_model.fit(np.delete(x_tmp, i, 0), np.delete(y_tmp, i), seed=seed)
        alphas[i] = np.float64(new_model.compute_loss(x, y).cpu().detach())

    pval = sum(alphas >= alphas[-1]) / (N + 1)

    return pval, list(alphas)


def ordinary_full_CP(model, x_train, y_train, xtest, yhat, labels=None, seed=42):
    """Runs ordinary full CP to make a prediction for (xtest, yhat)."""
    N = len(x_train)
    x_tmp = np.row_stack((x_train, [xtest]))
    y_tmp = np.concatenate((y_train, [yhat]))

    # Train model on full data.
    model.fit(x_tmp, y_tmp, seed=seed)
    alphas = [
        np.float64(model.compute_loss(x_tmp, y_tmp).cpu().detach())
        for x_tmp, y_tmp in zip(x_tmp, y_tmp)
    ]

    pval = sum(alphas >= alphas[-1]) / (N + 1)

    return pval, list(alphas)


def split_cp(
    model,
    x_train,
    y_train,
    x_test,
    labels=None,
    out_file=None,
    validation_split=0.2,
    seed=42,
):
    """Runs split/inductive CP for all points in x_test"""

    # Split data in training and calibration set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=seed
    )
    # Train model on training set
    model.fit(x_train, y_train, seed=seed)

    # Calculate non-conformity scores in calibration set
    alphas = [
        np.float64(model.compute_loss(xval, yval).cpu().detach())
        for xval, yval in zip(x_val, y_val)
    ]
    pvals = []
    prediction_times = []

    for _, xtest in enumerate(x_test):
        pvals_xtest = {}
        start = time.perf_counter()

        if labels is not None:
            for yhat in labels:
                alpha_test = np.float64(model.compute_loss(xtest, yhat).cpu().detach())
                pval = (sum(alphas >= alpha_test) + 1) / (len(x_val) + 1)
                pvals_xtest[yhat] = pval

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


def RAPS(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    labels=None,
    out_file=None,
    eps=None,
    seed=42,
    save_results=True,
):
    """Runs RAPS for all points in x_test (Angelopoulos et al.)"""
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )
    prediction_times = {}

    start = time.perf_counter()
    model.fit(x_train, y_train, seed=seed)
    calib_data = create_set(x_val, y_val)
    calib_loader = DataLoader(calib_data, batch_size=128, shuffle=True, pin_memory=True)  # type: ignore
    prediction_times["Training"] = [time.perf_counter() - start]

    if eps is None:
        epsilons = np.arange(0, 1.01, 0.01)
    else:
        epsilons = [eps]

    sizes = {}
    sets = {}

    for _, epsilon in enumerate(epsilons):
        cmodel = ConformalModel(
            model,
            calib_loader,
            alpha=epsilon,
            lamda_criterion="size",
            allow_zero_sets=True,
        )
        for j in range(len(y_test)):
            start = time.perf_counter()
            _, prediction_set = cmodel(check_tensor(np.expand_dims(x_test[j], axis=0)).to(device))  # type: ignore

            if epsilon not in sets:
                sets[epsilon] = [list(prediction_set[0])]
                sizes[epsilon] = [len(list(prediction_set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(prediction_set[0]))
                sizes[epsilon].append(len(list(prediction_set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)

    if save_results and out_file is not None:
        a_file = open(out_file + "_all_sets.pkl", "wb")
        pickle.dump(sets, a_file)
        a_file.close()

        a_file = open(out_file + "_all_sizes.pkl", "wb")
        pickle.dump(sizes, a_file)
        a_file.close()

        a_file = open(out_file + "_all_times.pkl", "wb")
        pickle.dump(prediction_times, a_file)
        a_file.close()

    return


def APS(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    labels=None,
    seed=42,
    out_file=None,
    eps=None,
    save_results=True,
):
    """Runs APS for all points in x_test (Romano et al.)"""
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )
    prediction_times = {}

    start = time.perf_counter()
    model.fit(x_train, y_train, seed=seed)
    prediction_times["Training"] = [time.perf_counter() - start]

    if eps is None:
        epsilons = np.arange(0, 1.01, 0.01)
    else:
        epsilons = [eps]

    sizes = {}
    sets = {}

    for _, epsilon in enumerate(epsilons):
        method_sc = methods.SplitConformal(
            x_val, y_val, model, alpha=epsilon, seed=seed, verbose=True
        )
        for j in range(len(y_test)):
            start = time.perf_counter()
            prediction_set = method_sc.predict(
                check_tensor(np.expand_dims(x_test[j], axis=0)).to(device)  # type: ignore
            )

            if epsilon not in sets:
                sets[epsilon] = [list(prediction_set[0])]
                sizes[epsilon] = [len(list(prediction_set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(prediction_set[0]))
                sizes[epsilon].append(len(list(prediction_set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)

    if save_results and out_file is not None:
        a_file = open(out_file + "_all_sets.pkl", "wb")
        pickle.dump(sets, a_file)
        a_file.close()

        a_file = open(out_file + "_all_sizes.pkl", "wb")
        pickle.dump(sizes, a_file)
        a_file.close()

        a_file = open(out_file + "_all_times.pkl", "wb")
        pickle.dump(prediction_times, a_file)
        a_file.close()

    return


def CV_plus(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    labels=None,
    seed=42,
    out_file=None,
    eps=None,
    save_results=True,
):
    """Runs CV+ for all points in x_test (Romano et al.)"""

    prediction_times = {}

    start = time.perf_counter()
    method_cv = methods.CVPlus(x_train, y_train, model, n_folds=5, seed=seed, verbose=False)
    prediction_times["Training"] = [time.perf_counter() - start]

    if eps is None:
        epsilons = np.arange(0, 1.01, 0.01)
    else:
        epsilons = [eps]

    sizes = {}
    sets = {}

    for _, epsilon in enumerate(epsilons):
        for j in range(len(y_test)):
            start = time.perf_counter()
            prediction_set = method_cv.predict(
                check_tensor(np.expand_dims(x_test[j], axis=0)).to(device), alpha=epsilon  # type: ignore
            )

            if epsilon not in sets:
                sets[epsilon] = [list(prediction_set[0])]
                sizes[epsilon] = [len(list(prediction_set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(prediction_set[0]))
                sizes[epsilon].append(len(list(prediction_set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)

    if save_results and out_file is not None:
        a_file = open(out_file + "_all_sets.pkl", "wb")
        pickle.dump(sets, a_file)
        a_file.close()

        a_file = open(out_file + "_all_sizes.pkl", "wb")
        pickle.dump(sizes, a_file)
        a_file.close()

        a_file = open(out_file + "_all_times.pkl", "wb")
        pickle.dump(prediction_times, a_file)
        a_file.close()

    return


def JK_plus(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    labels=None,
    seed=42,
    out_file=None,
    eps=None,
    save_results=True,
):
    """Runs JK+ for all points in x_test (Romano et al.)"""
    prediction_times = {}

    start = time.perf_counter()
    method_jk = methods.JackknifePlus(x_train, y_train, model, seed=seed, verbose=True)
    prediction_times["Training"] = [time.perf_counter() - start]

    if eps is None:
        epsilons = np.arange(0, 1.01, 0.01)
    else:
        epsilons = [eps]

    sizes = {}
    sets = {}

    for _, epsilon in enumerate(epsilons):
        for j in range(len(y_test)):
            start = time.perf_counter()
            prediction_set = method_jk.predict(
                check_tensor(np.expand_dims(x_test[j], axis=0)).to(device), alpha=epsilon  # type: ignore
            )

            if epsilon not in sets:
                sets[epsilon] = [list(prediction_set[0])]
                sizes[epsilon] = [len(list(prediction_set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(prediction_set[0]))
                sizes[epsilon].append(len(list(prediction_set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)

    if save_results and out_file is not None:
        a_file = open(out_file + "_all_sets.pkl", "wb")
        pickle.dump(sets, a_file)
        a_file.close()

        a_file = open(out_file + "_all_sizes.pkl", "wb")
        pickle.dump(sizes, a_file)
        a_file.close()

        a_file = open(out_file + "_all_times.pkl", "wb")
        pickle.dump(prediction_times, a_file)
        a_file.close()

    return
