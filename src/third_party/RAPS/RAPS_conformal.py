import os
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as transforms
from scipy.special import softmax
from tqdm import tqdm

device = torch.device("cuda")


# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.
class ConformalModel(nn.Module):
    def __init__(
        self,
        model,
        calib_loader,
        alpha,
        kreg=None,
        lamda=None,
        randomized=True,
        allow_zero_sets=False,
        pct_paramtune=0.3,
        batch_size=32,
        lamda_criterion="size",
    ):
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.T = torch.Tensor([1.3])  # initialize (1.3 is usually a good value)
        self.T, calib_logits = platt(self, calib_loader)
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = len(
            np.unique([calib_loader.dataset[i][1] for i in range(len(calib_loader.dataset))])
        )
        # self.num_classes = 10

        if kreg == None or lamda == None:
            kreg, lamda, calib_logits = pick_parameters(
                model,
                calib_logits,
                alpha,
                kreg,
                lamda,
                randomized,
                allow_zero_sets,
                pct_paramtune,
                batch_size,
                lamda_criterion,
            )

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda

        calib_loader = tdata.DataLoader(
            calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(
                scores,
                self.Qhat,
                I=I,
                ordered=ordered,
                cumsum=cumsum,
                penalties=self.penalties,
                randomized=randomized,
                allow_zero_sets=allow_zero_sets,
            )

        return logits, S


# Computes the conformal calibration
def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            logits = cmodel.model(x.float().to(device)).detach().cpu().numpy()
            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate(
                (
                    E,
                    giq(
                        scores,
                        targets,
                        I=I,
                        ordered=ordered,
                        cumsum=cumsum,
                        penalties=cmodel.penalties,
                        randomized=True,
                        allow_zero_sets=True,
                    ),
                )
            )

        Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation="higher")

        return Qhat


# Temperature scaling
def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(
        logits_dataset, batch_size=calib_loader.batch_size, shuffle=False, pin_memory=True
    )

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset


"""


        INTERNAL FUNCTIONS


"""

### Precomputed-logit versions of the above functions.


class ConformalModelLogits(nn.Module):
    def __init__(
        self,
        model,
        calib_loader,
        alpha,
        kreg=None,
        lamda=None,
        randomized=True,
        allow_zero_sets=False,
        naive=False,
        LAC=False,
        pct_paramtune=0.3,
        batch_size=32,
        lamda_criterion="size",
    ):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)

        if (kreg == None or lamda == None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(
                model,
                calib_loader.dataset,
                alpha,
                kreg,
                lamda,
                randomized,
                allow_zero_sets,
                pct_paramtune,
                batch_size,
                lamda_criterion,
            )
            calib_loader = tdata.DataLoader(
                calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True
            )

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1 - alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array(
                [
                    np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0]
                    for x in calib_loader.dataset
                ]
            )
            scores_cal = 1 - np.array(
                [
                    np.sort(torch.softmax(calib_loader.dataset[i][0] / self.T.item(), dim=0))[
                        ::-1
                    ][gt_locs_cal[i]]
                    for i in range(len(calib_loader.dataset))
                ]
            )
            self.Qhat = np.quantile(
                scores_cal, np.ceil((scores_cal.shape[0] + 1) * (1 - alpha)) / scores_cal.shape[0]
            )

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)

                S = gcq(
                    scores,
                    self.Qhat,
                    I=I,
                    ordered=ordered,
                    cumsum=cumsum,
                    penalties=self.penalties,
                    randomized=randomized,
                    allow_zero_sets=allow_zero_sets,
                )
            else:
                S = [np.where((1 - scores[i, :]) < self.Qhat)[0] for i in range(scores.shape[0])]

        return logits, S


def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate(
                (
                    E,
                    giq(
                        scores,
                        targets,
                        I=I,
                        ordered=ordered,
                        cumsum=cumsum,
                        penalties=cmodel.penalties,
                        randomized=True,
                        allow_zero_sets=True,
                    ),
                )
            )

        Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation="higher")

        return Qhat


def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().to(device)

    T = nn.Parameter(torch.Tensor([1.3]).to(device))

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.to(device)
            x.requires_grad = True
            out = x / T
            loss = nll_criterion(out, targets.long().to(device))
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T


### CORE CONFORMAL INFERENCE FUNCTIONS


# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = (
                1
                / ordered[i, sizes_base[i] - 1]
                * (
                    tau
                    - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1])
                    - penalties_cumsum[0, sizes_base[i] - 1]
                )
            )  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[
            1
        ]  # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[
            sizes == 0
        ] = 1  # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [
            I[i, 0 : sizes[i]],
        ]

    return S


# Get the 'p-value'
def get_tau(
    score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets
):  # For one example
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return (
            U * ordered[idx] + cumsum[(idx[0], idx[1] - 1)] + (penalty[0 : (idx[1][0] + 1)]).sum()
        )


# Gets the histogram of Taus.
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
    Generalized inverse quantile conformity score function.
    E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(
            scores[i : i + 1, :],
            targets[i].item(),
            I[i : i + 1, :],
            ordered[i : i + 1, :],
            cumsum[i : i + 1, :],
            penalties[0, :],
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )

    return E


### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array(
        [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits]
    )
    kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation="higher") + 1
    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [
        0.001,
        0.01,
        0.1,
        0.2,
        0.5,
    ]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
        )
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(
            paramtune_loader, conformal_model, print_bool=False
        )
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star


def pick_lamda_adaptiveness(
    model,
    paramtune_loader,
    alpha,
    kreg,
    randomized,
    allow_zero_sets,
    strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]],
):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [
        0,
        1e-5,
        1e-4,
        8e-4,
        9e-4,
        1e-3,
        1.5e-3,
        2e-3,
    ]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
        )
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


def pick_parameters(
    model,
    calib_logits,
    alpha,
    kreg,
    lamda,
    randomized,
    allow_zero_sets,
    pct_paramtune,
    batch_size,
    lamda_criterion,
):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(
        calib_logits, [num_paramtune, len(calib_logits) - num_paramtune]
    )
    calib_loader = tdata.DataLoader(
        calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    paramtune_loader = tdata.DataLoader(
        paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
    return kreg, lamda, calib_logits


def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=["size", "correct"])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(
            logit.float()
        )  # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int(target[j] in list(S[j]))
        batch_df = pd.DataFrame({"size": size, "correct": correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df["size"] >= stratum[0]) & (df["size"] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation  # the violation


def sort_sum(scores):
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter("batch_time")
        # top1 = AverageMeter('top1')
        # top5 = AverageMeter('top5')
        coverage = AverageMeter("RAPS coverage")
        size = AverageMeter("RAPS size")
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.to(device)
            # compute output
            output, S = model(x.float().to(device))
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            # top1.update(prec1.item()/100.0, n=x.shape[0])
            # top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            # if print_bool:
            #    print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print("")  # Endline

    return 0, 0, coverage.avg, size.avg


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).to(device)
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets


def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset) - n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset) - n1 - n2])
    return data1, data2


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0] - n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0] - n1 - n2])
    return data1, data2


def get_model(modelname):
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=True, progress=True)

    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True, progress=True)

    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True, progress=True)

    elif modelname == "ResNeXt101":
        model = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)

    elif modelname == "VGG16":
        model = torchvision.models.vgg16(pretrained=True, progress=True)

    elif modelname == "ShuffleNet":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True, progress=True)

    elif modelname == "Inception":
        model = torchvision.models.inception_v3(pretrained=True, progress=True)

    elif modelname == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True, progress=True)

    else:
        raise NotImplementedError

    model.eval()
    model = torch.nn.DataParallel(model).to(device)

    return model


# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    logits = torch.zeros(
        (
            len(loader.dataset),
            len(np.unique([loader.dataset[i][1] for i in range(len(loader.dataset))])),
        )
    )  # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f"Computing logits for model (only happens once).")
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.float().to(device)).detach().cpu()
            logits[i : (i + x.shape[0]), :] = batch_logits
            labels[i : (i + x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits


def get_logits_dataset(
    modelname,
    datasetname,
    datasetpath,
    cache=str(pathlib.Path(__file__).parent.absolute()) + "/experiments/.cache/",
):
    fname = cache + datasetname + "/" + modelname + ".pkl"

    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, "rb") as handle:
            return pickle.load(handle)

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(modelname)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    # Save the dataset
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits
