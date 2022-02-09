import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
import sys
from tqdm import tqdm
import copy

from third_party.ARC.arc.classification import ProbabilityAccumulator as ProbAccum

class CVPlus:
    def __init__(self, X, Y, black_box, n_folds=5, seed=1000, verbose=False):
        X = np.array(X)
        Y = np.array(Y)
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.n_classes = len(self.classes)
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        self.verbose = verbose
        
        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = []

        for train_index, _ in self.cv.split(X):
            model = copy.deepcopy(black_box)
            model.fit(X[train_index], Y[train_index], seed = seed)
            self.mu_LOO.append(model)

        # Accumulate probabilities for the original data with the grey boxes
        test_indices = [test_index for _, test_index in self.cv.split(X)]
        self.test_indices = test_indices
        self.folds = [[]]*self.n
        for k in range(self.n_folds):
            for i in test_indices[k]:
                self.folds[i] = k
        self.grey_boxes = [[]]*self.n_folds
        if self.verbose:
            print("Training black boxes on {} samples with {}-fold cross-validation:".
                  format(self.n, self.n_folds), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]).cpu().detach())
        else:
            for k in range(self.n_folds):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]).cpu().detach())
               
        # Compute scores using real labels
        epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.alpha_max = np.zeros((self.n, 1))
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
        else:
            for k in range(self.n_folds):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
            
    def predict(self, X, alpha):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))

        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for fold in tqdm(range(self.n_folds), ascii=True, disable=True):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X).cpu().detach())
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for fold in range(self.n_folds):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X).cpu().detach())
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])

        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
                
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(self.n))
        S = [ np.where(prop_smaller[i,:] < level_adjusted)[0] for i in range(n) ]
        return S

class JackknifePlus:
    def __init__(self, X, Y, black_box, seed=1000, verbose=True):
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.verbose = verbose

        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [[]] * self.n
        if self.verbose:
            print("Training black boxes on {} samples with the Jacknife+:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                model = copy.deepcopy(black_box)
                model.fit(np.delete(X,i,0), np.delete(Y,i), seed = seed)
                self.mu_LOO[i] = model
        else:
            for i in range(self.n):
                print(i)
                model = copy.deepcopy(black_box)
                model.fit(np.delete(X,i,0), np.delete(Y,i), seed = seed)
                self.mu_LOO[i] = model

        # Accumulate probabilities for the original data with the grey boxes
        self.grey_boxes = [ ProbAccum(self.mu_LOO[i].predict_proba(X[i]).cpu().detach()) for i in range(self.n) ]

        # Compute scores using real labels
        epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        
        self.alpha_max = np.zeros((self.n, 1))    
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
        else:
            for i in range(self.n):
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
                
    def predict(self, X, alpha):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))
        
        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X).cpu().detach())
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for i in range(self.n):
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X).cpu().detach())
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
                
        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(self.n))
        S = [ np.where(prop_smaller[i,:] < level_adjusted)[0] for i in range(n) ]
        return S

class SplitConformal:
    def __init__(self, X_calib, Y_calib, black_box, alpha, seed=1000, verbose=False):
        # Split data into training/calibration sets
        
        self.black_box = black_box

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib).cpu().detach()
        grey_box = ProbAccum(p_hat_calib)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=len(Y_calib))
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(len(Y_calib)))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

    def predict(self, X):
        n = X.shape[0]
        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X).cpu().detach()
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon)
        return S_hat
