"""Author: Javier Abad Martinez"""

import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.autograd import grad
from torch.nn import CrossEntropyLoss

from acp.models import NeuralNetworkTemplate
from acp.utils import check_tensor, flatten_gradient, log_to_file

# pylint: disable=no-member
# no error

device = torch.device("cuda")


class ACP:
    """
    ACP class template
    """

    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        model: NeuralNetworkTemplate,
        seed: int = 42,
        verbose: bool = False,
        batches: int = 1,
        damp: float = 0.001,
    ):
        """
        ACP outputs a prediction set that contains the true label with at least a probability
        specified by the practicioner.

        When instantiated, the ACP class runs the training step, where it computes the gradient
        and nonconformity score at each training sample and the Hessian inverse.

        Args:
            x_train (Tensor): input features.
            y_train (Tensor): input labels.
            model (nn.Module): PyTorch model wrapped by ACP, with .fit() and .predict() methods.
            seed (int, optional): random seed. Defaults to 42.
            verbose (bool, optional): verbose. Defaults to False.
            batches (int, optional): number of batches to compute the Hessian. Defaults to 1.
            damp (float, optional): damping parameter for the Hessian. Defaults to 0.001.
        """
        self.seed = seed
        self.verbose = verbose
        self.model = model
        self.batches = batches
        self.n_samples = len(x_train)
        self.labels = np.unique(y_train).tolist()

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        # Train model on (x_train, y_train).
        self.model = self.model.to(device)
        if self.verbose:
            print(f"Training model on {self.n_samples} samples")
        self.model.fit(x_train, y_train, seed=self.seed)
        torch.cuda.empty_cache()

        # Estimate influence.
        if self.verbose:
            print("Computing gradients and Hessian inverse")
        self.gradients = []
        for x, y in zip(x_train, y_train):
            self.gradients.append(self.compute_gradient(x, y).cpu().detach())  # type: ignore
            torch.cuda.empty_cache()
        self.gradients.append(None)  # for the test point

        hessian = torch.zeros(self.count_params(), self.count_params()).to(device)
        x_train_splitted = np.array_split(x_train, self.batches)
        y_train_splitted = np.array_split(y_train, self.batches)
        for batch_x, batch_y in zip(x_train_splitted, y_train_splitted):
            hessian += self.compute_hessian(batch_x, batch_y)
        hessian = hessian / self.batches
        hessian += torch.diag(Tensor([damp] * len(hessian))).to(device)
        torch.cuda.empty_cache()
        self.hessian_inv = torch.inverse(hessian)
        del hessian

        # Compute preliminary scores.
        if verbose:
            print(f"Computing scores for {self.n_samples} samples")
        self.losses = [
            np.float64(self.compute_loss(x, y).cpu().detach()) for x, y in zip(x_train, y_train)
        ]
        self.losses.append(None)  # type: ignore # for the test point

        if self.verbose:
            print("Conformal predictor is ready!")

    def loss(self, targets: Tensor, outputs: Tensor) -> Tensor:
        """
        Method to compute the cross-entropy loss between predictions and targets.

        Args:
            targets (Tensor): groun truth.
            outputs (Tensor): predictions.

        Returns:
            Tensor: loss
        """

        targets = targets.long()
        loss_func = CrossEntropyLoss(reduction="mean")
        loss = loss_func(outputs, targets)
        return loss

    def compute_loss(self, x: Tensor, y: Tensor, gpu: int = 0) -> Tensor:
        """
        Method to compute the model loss at an input tensor.

        Args:
            x (Tensor): input features.
            y (Tensor): input labels.
            gpu (int, optional): use cpu or gpu. Defaults to 0.

        Returns:
            Tensor: loss at inputs
        """

        self.model.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(y).unsqueeze(0)
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self.model(x)
        return self.loss(targets, outputs)

    def compute_gradient(self, x: Tensor, targets: Tensor, gpu: int = 0, flatten: bool = True):
        """
        Method to compute the gradient of the loss wrt the model parameters.

        Args:
            x (Tensor): input features.
            targets (Tensor): targets/ground truth.
            gpu (int, optional): use cpu or gpu. Defaults to 0.
            flatten (bool, optional): flatten output vector. Defaults to True.

        Returns:
            Sequence[Tensor]: gradients of the loss at input vector wrt the model parameters.
        """
        self.model.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(targets).unsqueeze(0)
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self.model(x)
        loss = self.loss(targets, outputs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        gradient = list(grad(loss, params, create_graph=True))
        if flatten:
            return flatten_gradient(gradient)
        return gradient

    def compute_hessian(self, x_train, y_train, gpu: int = 0) -> Tensor:
        """_
        Method to compute the Hessian of the loss wrt the model parameters.
        Args:
            x_train (_type_): input features.
            y_train (_type_): targets/ground truth
            gpu (int, optional): use cpu or gpu. Defaults to 0.

        Returns:
            Tensor: Hessian of the loss at input vector wrt the model parameters.
        """

        def gradient(
            outputs: Tensor,
            inputs: Sequence[Tensor],
            grad_outputs: Optional[Sequence[Tensor]] = None,
            retain_graph: Optional[bool] = None,
            create_graph: bool = False,
        ) -> Tensor:
            grads = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs,
                allow_unused=True,
                retain_graph=retain_graph,
                create_graph=create_graph,
            )
            grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
            return torch.cat([x.contiguous().view(-1) for x in grads])

        x_train = check_tensor(x_train)
        y_train = check_tensor(y_train)

        if gpu >= 0:
            x_train, y_train = x_train.to(device), y_train.to(device)

        outputs = self.model(x_train)
        loss = self.loss(y_train, outputs)
        inputs = list(self.model.parameters())
        hessian = torch.zeros(self.count_params(), self.count_params())

        row_index = 0

        for i, inp in enumerate(inputs):
            [gradd] = torch.autograd.grad(loss, inp, create_graph=True, allow_unused=False)
            gradd = torch.zeros_like(inp) if gradd is None else gradd
            gradd = gradd.contiguous().view(-1)

            for j in range(inp.numel()):
                if gradd[j].requires_grad:
                    row = gradient(gradd[j], iter(inputs[i:]), retain_graph=True, create_graph=False)[j:]  # type: ignore
                else:
                    row = gradd[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)  # type: ignore

                hessian[row_index, row_index:].add_(row.type_as(hessian))  # row_index's row
                if row_index + 1 < self.count_params():
                    hessian[row_index + 1 :, row_index].add_(
                        row[1:].type_as(hessian)
                    )  # row_index's column
                del row
                row_index += 1
            del gradd
        return Tensor(hessian).to(device)

    def count_params(self):
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())


class ACP_D(ACP):
    """Deleted ACP scheme (ACP (D) in the paper)."""

    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        model: NeuralNetworkTemplate,
        seed: int = 42,
        verbose: bool = False,
        batches: int = 1,
        damp: float = 0.001,
    ):
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            model=model,
            seed=seed,
            verbose=verbose,
            batches=batches,
            damp=damp,
        )

    def predict(self, x_test: Tensor, epsilon: float, out_file: Optional[str] = None) -> Dict:
        """
        Prediction step. ACP computes the gradient at the test point and uses influence
        functions to build a prediction set that contains the true label with a specified
        probability 1 - epsilon.

        Args:
            x_test (Tensor): input features.
            epsilon (int): significance level.
            out_file (Optional[str], optional): path to save results. Defaults to None.

        Returns:
            Tensor: prediction sets.
        """
        pred_sets = {}

        if self.verbose:
            print(f"Computing p-values for {len(x_test)} samples")

        for k, x_test in enumerate(tqdm.tqdm(x_test)):
            pvals_x_test = {}
            pred_set_x_test = []

            for yhat in self.labels:
                alphas = np.zeros(self.n_samples + 1)
                # Obtain gradient on test point
                g_test = self.compute_gradient(x_test, yhat, flatten=True)
                # Obtain loss on test point
                loss_test = np.float64(self.compute_loss(x_test, yhat).cpu().detach())

                self.gradients[-1] = g_test
                self.losses[-1] = loss_test

                for j, (loss, gradient) in enumerate(zip(self.losses, self.gradients)):
                    gradient = gradient.to(device)
                    # Compute influence
                    est = -gradient.T @ self.hessian_inv @ (g_test - gradient) / self.n_samples
                    # Compute nonconf. score
                    alphas[j] = loss + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()

                # Compute p-value
                pval = sum(alphas >= alphas[-1]) / (self.n_samples + 1)
                pvals_x_test[yhat] = pval

                # Check if yhat is included in prediction set for the given epsilon
                if epsilon < pval:
                    pred_set_x_test.append(yhat)

            if out_file:
                log_to_file(out_file, {"N": self.n_samples, "p-values": pvals_x_test})

            pred_sets[k] = pred_set_x_test

        return pred_sets


class ACP_O(ACP):
    """Ordinary ACP scheme (ACP (O) in the paper)."""

    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        model: NeuralNetworkTemplate,
        seed: int = 42,
        verbose: bool = False,
        batches: int = 1,
        damp: float = 0.001,
    ):
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            model=model,
            seed=seed,
            verbose=verbose,
            batches=batches,
            damp=damp,
        )

    def predict(self, x_test: Tensor, epsilon: float, out_file: Optional[str] = None) -> Dict:
        """
        Prediction step. ACP computes the gradient at the test point and uses influence
        functions to build a prediction set that contains the true label with a specified
        probability 1 - epsilon.

        Args:
            x_test (Tensor): input features.
            epsilon (int): significance level.
            out_file (Optional[str], optional): path to save results. Defaults to None.

        Returns:
            Tensor: prediction sets.
        """
        pred_sets = {}

        if self.verbose:
            print(f"Computing p-values for {len(x_test)} samples")

        for k, x_test in enumerate(tqdm.tqdm(x_test)):
            pvals_x_test = {}
            pred_set_x_test = []

            for yhat in self.labels:
                alphas = np.zeros(self.n_samples + 1)
                # Obtain gradient on test point
                g_test = self.compute_gradient(x_test, yhat, flatten=True)
                # Obtain loss on test point
                loss_test = np.float64(self.compute_loss(x_test, yhat).cpu().detach())

                self.gradients[-1] = g_test
                self.losses[-1] = loss_test

                for j, (loss, gradient) in enumerate(zip(self.losses, self.gradients)):
                    gradient = gradient.to(device)
                    # Compute influence
                    est = -gradient.T @ self.hessian_inv @ g_test / self.n_samples
                    # Compute nonconf. score
                    alphas[j] = loss + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()

                # Compute p-value
                pval = sum(alphas >= alphas[-1]) / (self.n_samples + 1)
                pvals_x_test[yhat] = pval

                # Check if yhat is included in prediction set for the given epsilon
                if epsilon < pval:
                    pred_set_x_test.append(yhat)

            if out_file:
                log_to_file(out_file, {"N": self.n_samples, "p-values": pvals_x_test})

            pred_sets[k] = pred_set_x_test

        return pred_sets
