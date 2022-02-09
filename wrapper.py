from utils import *
from models import *
import copy
import numpy as np
import torch 
import random
from tqdm import tqdm

class ACP:
    def __init__(self, Xtrain, Ytrain, model, seed = 42, verbose = False, batches = 1, damp = 0.001):
        
        self.seed = seed
        self.verbose = verbose   
        self.model = model
        self.batches = batches
        self.N = len(Xtrain)
        self.labels = np.unique(Ytrain).tolist()
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        
        # Train model on (Xtrain, Ytrain).
        self.model = self.model.to(device)
        if self.verbose:
            print("Training model on {} samples".format(self.N))
        self.model.fit(Xtrain, Ytrain, seed = self.seed)
        torch.cuda.empty_cache() 
        
        # Estimate influence.
        if self.verbose:
            print("Computing gradients and Hessian inverse")
        self.gradients = []
        for x, y in zip(Xtrain, Ytrain):
            self.gradients.append(self.model.grad_z(x, y, flatten = True).cpu().detach())
            torch.cuda.empty_cache()
        self.gradients.append(None) #for the test point
        
        H = torch.zeros(self.model.count_params(), self.model.count_params()).to(device)
        Xtrain_splitted = np.array_split(Xtrain, self.batches)
        Ytrain_splitted = np.array_split(Ytrain, self.batches)
        for batch_X, batch_Y in zip(Xtrain_splitted, Ytrain_splitted):
            H += self.model.hessian_all_points(batch_X, batch_Y)
        H = H/self.batches
        H += torch.diag(Tensor([damp]*len(H))).to(device)
        torch.cuda.empty_cache()
        self.H_inv = torch.inverse(H)
        del H 
        
        # Compute preliminary scores.
        if verbose:
            print("Computing scores for {} samples".format(self.N))
        self.losses = [np.float64(self.model.compute_loss(x, y).cpu().detach()) for x, y in zip(Xtrain, Ytrain)]
        self.losses.append(None) #for the test point
        
        if self.verbose:
            print("Conformal predictor is ready!")
            
    def predict(self, Xtest):
        "Predict method for CP function"
        return
        
class ACP_D(ACP):
            
    def __init__(self, Xtrain, Ytrain, model, seed = 42, verbose = False, batches = 1, damp = 0.001):
        super().__init__(Xtrain = Xtrain, Ytrain = Ytrain, model = model, seed = seed, verbose = verbose, batches = batches, damp = damp)
        
    def predict(self, Xtest, epsilon, out_file = None):
          
        pred_sets = {}
        
        if self.verbose:
                print("Computing p-values for {} samples".format(len(Xtest)))
                      
        for k, xtest in enumerate(tqdm(Xtest)):
                
            pvals_xtest = {}
            pred_set_xtest = []
                      
            for yhat in self.labels:
                
                alphas = np.zeros(self.N + 1)
                # Obtain gradient on test point
                g_test = self.model.grad_z(xtest, yhat, flatten = True)
                # Obtain loss on test point
                loss_test = np.float64(self.model.compute_loss(xtest, yhat).cpu().detach())
                      
                self.gradients[-1] = g_test
                self.losses[-1] = loss_test
                   
                for j, (loss, gradient) in enumerate(zip(self.losses, self.gradients)):
                    
                    gradient = gradient.to(device)
                    # Compute influence
                    est = - gradient.T@self.H_inv@(g_test-gradient)/self.N
                    #Compute nonconf. score
                    alphas[j] = loss + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()
                
                #Compute p-value
                pval = sum(alphas >= alphas[-1])/(self.N+1)
                pvals_xtest[yhat] = pval
                
                #Check if yhat is included in prediction set for the given epsilon
                if epsilon < pval:
                      pred_set_xtest.append(yhat)

            if out_file:
                log_to_file(out_file, {"N": self.N,
                                       "p-values": pvals_xtest
                                      })
        
            pred_sets[k] = pred_set_xtest
        
        return pred_sets
    
class ACP_O(ACP):
            
    def __init__(self, Xtrain, Ytrain, model, seed = 42, verbose = False, batches = 1, damp = 0.001):
        super().__init__(Xtrain = Xtrain, Ytrain = Ytrain, model = model, seed = seed, verbose = verbose, batches = batches, damp = damp)
        
    def predict(self, Xtest, epsilon, out_file = None):
          
        pred_sets = {}
        
        if self.verbose:
                print("Computing p-values for {} samples".format(len(Xtest)))
                      
        for k, xtest in enumerate(tqdm(Xtest)):
                      
            pvals_xtest = {}
            pred_set_xtest = []
                      
            for yhat in self.labels:
                     
                alphas = np.zeros(self.N + 1)
                # Obtain gradient on test point
                g_test = self.model.grad_z(xtest, yhat, flatten = True)
                # Obtain loss on test point
                loss_test = np.float64(self.model.compute_loss(xtest, yhat).cpu().detach())
                      
                self.gradients[-1] = g_test
                self.losses[-1] = loss_test
                   
                for j, (loss, gradient) in enumerate(zip(self.losses, self.gradients)):
                    
                    gradient = gradient.to(device)
                    # Compute influence
                    est = - gradient.T@self.H_inv@g_test/self.N
                    #Compute nonconf. score
                    alphas[j] = loss + np.array(est.cpu().detach())
                    torch.cuda.empty_cache()
                
                #Compute p-value
                pval = sum(alphas >= alphas[-1])/(self.N+1)
                pvals_xtest[yhat] = pval
                
                #Check if yhat is included in prediction set for the given epsilon
                if epsilon < pval:
                      pred_set_xtest.append(yhat)

            if out_file:
                log_to_file(out_file, {"N": self.N,
                                       "p-values": pvals_xtest
                                      })
        
            pred_sets[k] = pred_set_xtest
        
        return pred_sets
        
        
        
            
    
    