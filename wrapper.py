from utils import *
from models import *
import copy
import numpy as np
import torch 
import random
from tqdm import tqdm
from torch.autograd import grad

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
            self.gradients.append(self.compute_gradient(x, y).cpu().detach())
            torch.cuda.empty_cache()
        self.gradients.append(None) #for the test point
        
        H = torch.zeros(self.count_params(), self.count_params()).to(device)
        Xtrain_splitted = np.array_split(Xtrain, self.batches)
        Ytrain_splitted = np.array_split(Ytrain, self.batches)
        for batch_X, batch_Y in zip(Xtrain_splitted, Ytrain_splitted):
            H += self.compute_hessian(batch_X, batch_Y)
        H = H/self.batches
        H += torch.diag(Tensor([damp]*len(H))).to(device)
        torch.cuda.empty_cache()
        self.H_inv = torch.inverse(H)
        del H 
        
        # Compute preliminary scores.
        if verbose:
            print("Computing scores for {} samples".format(self.N))
        self.losses = [np.float64(self.compute_loss(x, y).cpu().detach()) for x, y in zip(Xtrain, Ytrain)]
        self.losses.append(None) #for the test point
        
        if self.verbose:
            print("Conformal predictor is ready!")
    
    def loss(self, targets, outputs):
        targets = targets.long()
        loss_func = CrossEntropyLoss(reduction = "mean")
        loss = loss_func(outputs, targets)      
        return loss
    
    def compute_loss(self, x, y, gpu = 0):
        self.model.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(y).unsqueeze(0)
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self.model(x)
        return self.loss(targets, outputs)
    
    def compute_gradient(self, x, targets, gpu = 0, flatten = True):
        self.model.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(targets).unsqueeze(0)
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self.model(x)
        loss = self.loss(targets, outputs)
        params = [ p for p in self.model.parameters() if p.requires_grad ]
        gradient = list(grad(loss, params, create_graph=True))
        if flatten:
            return flatten_gradient(gradient)           
        return gradient
    
    def compute_hessian(self, Xtrain, Ytrain, gpu = 0):
        
        def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):

            if torch.is_tensor(inputs):
                inputs = [inputs]
            else:
                inputs = list(inputs)
            grads = torch.autograd.grad(outputs, inputs, grad_outputs, 
                         allow_unused=True,retain_graph=retain_graph,
                         create_graph=create_graph)
            grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
            return torch.cat([x.contiguous().view(-1) for x in grads])
        
        Xtrain = check_tensor(Xtrain)
        Ytrain = check_tensor(Ytrain)
        
        if gpu >= 0:
            Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
            
        outputs = self.model(Xtrain)        
        loss = self.loss(Ytrain, outputs)        
        inputs = self.model.parameters()
        hessian = torch.zeros(self.count_params(), self.count_params())

        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)
            
        row_index = 0

        for i, inp in enumerate(inputs):

            [grad] = torch.autograd.grad(loss, inp, create_graph=True, allow_unused=False)
            grad = torch.zeros_like(inp) if grad is None else grad
            grad = grad.contiguous().view(-1)

            for j in range(inp.numel()):
                if grad[j].requires_grad:
                    row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=False)[j:]
                else:
                    row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

                hessian[row_index, row_index:].add_(row.type_as(hessian))  # row_index's row
                if row_index + 1 < self.count_params():
                    hessian[row_index + 1:, row_index].add_(row[1:].type_as(hessian))  # row_index's column
                del row
                row_index += 1
            del grad
        return Tensor(hessian).to(device)
    
    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
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
                g_test = self.compute_gradient(xtest, yhat, flatten = True)
                # Obtain loss on test point
                loss_test = np.float64(self.compute_loss(xtest, yhat).cpu().detach())
                      
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
                g_test = self.compute_gradient(xtest, yhat, flatten = True)
                # Obtain loss on test point
                loss_test = np.float64(self.compute_loss(xtest, yhat).cpu().detach())
                      
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
        
        
        
            
    
    