"""Author: Javier Abad Martinez"""


import time

import numpy as np
import torch
from torch import Tensor

from acp.utils import log_to_file

# pylint: disable=no-member
# no error

device = torch.device("cuda")


def ACP_D(
    model,
    x_train,
    y_train,
    x_test,
    labels=None,
    out_file=None,
    seed=42,
    damp=10**-3,
    batches=1,
):
    """Runs ACP (deleted) to make a prediction for all points in x_test. IFs for incremental and decremental learning"""
    N = len(x_train)
    # Train model on D.
    model_D = model
    model_D = model_D.to(device)
    model_D.fit(x_train, y_train, seed=seed)
    torch.cuda.empty_cache()
    # Estimate influence.
    gradients = []
    for x, y in zip(x_train, y_train):
        gradients.append(model_D.grad_z(x, y, flatten=True).cpu().detach())
        torch.cuda.empty_cache()
    gradients.append(None)  # for the test point

    H_D = torch.zeros(model_D.count_params(), model_D.count_params()).to(device)
    x_train_splitted = np.array_split(x_train, batches)
    y_train_splitted = np.array_split(y_train, batches)
    for batch_x, batch_y in zip(x_train_splitted, y_train_splitted):
        H_D += model_D.hessian_all_points(batch_x, batch_y)
    H_D = H_D / batches
    H_D += torch.diag(Tensor([damp] * len(H_D))).to(device)
    torch.cuda.empty_cache()
    H_inv = torch.inverse(H_D)
    del H_D

    # Preliminary scores
    losses = [
        np.float64(model_D.compute_loss(x, y).cpu().detach()) for x, y in zip(x_train, y_train)
    ]
    losses.append(None)  # type: ignore

    pvals = []
    prediction_times = []

    for k, xtest in enumerate(x_test):
        print(f"Test point: {k}")
        pvals_xtest = {}
        scores = {}
        start = time.perf_counter()
        if labels is not None:
            for yhat in labels:
                # Extended dataset
                x_tmp = np.row_stack((x_train, [xtest]))
                y_tmp = np.concatenate((y_train, [yhat]))
                alphas = np.zeros(len(x_tmp))
                # Obtain gradient on test point
                g_test = model_D.grad_z(x_tmp[-1, :], y_tmp[-1], flatten=True)
                # Obtain loss on test point
                loss_test = np.float64(
                    model_D.compute_loss(x_tmp[-1, :], y_tmp[-1]).cpu().detach()
                )
                gradients[-1] = g_test
                losses[-1] = loss_test

                for j, (x, y) in enumerate(zip(x_tmp, y_tmp)):
                    gradient = gradients[j].to(device)
                    # Compute influence
                    est = -gradient.T @ H_inv @ (g_test - gradient) / N
                    alphas[j] = losses[j] + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()

                pval = sum(alphas >= alphas[-1]) / (N + 1)
                pvals_xtest[yhat], scores[yhat] = pval, list(alphas)

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


def ACP_O(
    model,
    x_train,
    y_train,
    x_test,
    labels=None,
    out_file=None,
    seed=42,
    damp=10**-3,
    batches=1,
):
    """Runs ACP (ordinary) to make a prediction for all points in x_test"""

    N = len(x_train)
    # Train model on D.
    model_D = model
    model_D = model_D.to(device)
    model_D.fit(x_train, y_train, seed=seed)
    torch.cuda.empty_cache()
    # Estimate influence.
    gradients = []
    for x, y in zip(x_train, y_train):
        gradients.append(model_D.grad_z(x, y, flatten=True).cpu().detach())
        torch.cuda.empty_cache()
    gradients.append(None)  # for the test point

    H_D = torch.zeros(model_D.count_params(), model_D.count_params()).to(device)
    x_train_splitted = np.array_split(x_train, batches)
    y_train_splitted = np.array_split(y_train, batches)
    for batch_x, batch_y in zip(x_train_splitted, y_train_splitted):
        H_D += model_D.hessian_all_points(batch_x, batch_y)
    H_D = H_D / batches
    H_D += torch.diag(Tensor([damp] * len(H_D))).to(device)
    torch.cuda.empty_cache()
    H_inv = torch.inverse(H_D)
    del H_D

    # Preliminary scores
    losses = [
        np.float64(model_D.compute_loss(x, y).cpu().detach()) for x, y in zip(x_train, y_train)
    ]
    losses.append(None)  # type: ignore

    pvals = []
    prediction_times = []

    for k, xtest in enumerate(x_test):
        print(f"Test point: {k}")
        pvals_xtest = {}
        scores = {}
        start = time.perf_counter()
        if labels is not None:
            for yhat in labels:
                # Extended dataset
                x_tmp = np.row_stack((x_train, [xtest]))
                y_tmp = np.concatenate((y_train, [yhat]))
                alphas = np.zeros(len(x_tmp))
                # Obtain gradient on test point
                g_test = model_D.grad_z(x_tmp[-1, :], y_tmp[-1], flatten=True)
                # Obtain loss on test point
                loss_test = np.float64(
                    model_D.compute_loss(x_tmp[-1, :], y_tmp[-1]).cpu().detach()
                )
                gradients[-1] = g_test
                losses[-1] = loss_test

                for j, (x, y) in enumerate(zip(x_tmp, y_tmp)):
                    gradient = gradients[j].to(device)
                    # Compute influence
                    est = -gradient.T @ H_inv @ g_test / N
                    alphas[j] = losses[j] + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()

                pval = sum(alphas >= alphas[-1]) / (N + 1)
                pvals_xtest[yhat], scores[yhat] = pval, list(alphas)

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
