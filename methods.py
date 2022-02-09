from utils import *
from models import *
import copy

device = torch.device('cuda')

def ACP_D(model, Xtrain, Ytrain, Xtest, labels = [0,1], out_file = None, seed = 42, damp = 10**-3, batches = 1):
    
    """Runs ACP (deleted) to make a prediction for all points in Xtest. IFs for incremental and decremental learning 
    """
    N = len(Xtrain)
    # Train model on D.
    model_D = model
    model_D = model_D.to(device)
    model_D.fit(Xtrain, Ytrain, seed = seed)
    torch.cuda.empty_cache() 
    # Estimate influence.
    gradients = []
    for x, y in zip(Xtrain, Ytrain):
        gradients.append(model_D.grad_z(x, y, flatten = True).cpu().detach())
        torch.cuda.empty_cache()
    gradients.append(None) #for the test point
    
    H_D = torch.zeros(model_D.count_params(), model_D.count_params()).to(device)
    Xtrain_splitted = np.array_split(Xtrain, batches)
    Ytrain_splitted = np.array_split(Ytrain, batches)
    for batch_X, batch_Y in zip(Xtrain_splitted, Ytrain_splitted):
        H_D += model_D.hessian_all_points(batch_X, batch_Y)
    H_D = H_D/batches
    H_D += torch.diag(Tensor([damp]*len(H_D))).to(device)
    torch.cuda.empty_cache()
    H_inv = torch.inverse(H_D)
    del H_D
    
    #Preliminary scores
    losses = [np.float64(model_D.compute_loss(x, y).cpu().detach()) for x, y in zip(Xtrain, Ytrain)]
    losses.append(None)
    
    pvals = []
    prediction_times = []
    
    for k, xtest in enumerate(Xtest):
        print("TEST: " +str(k+1))
        pvals_xtest = {}
        scores = {}
        start = time.perf_counter()
        
        for yhat in labels:
            # Extended dataset
            Xtmp = np.row_stack((Xtrain, [xtest]))
            Ytmp = np.concatenate((Ytrain, [yhat]))
            alphas = np.zeros(len(Xtmp))
            # Obtain gradient on test point
            g_test = model_D.grad_z(Xtmp[-1,:], Ytmp[-1], flatten = True)
            # Obtain loss on test point
            loss_test = np.float64(model_D.compute_loss(Xtmp[-1,:], Ytmp[-1]).cpu().detach())
            gradients[-1] = g_test
            losses[-1] = loss_test

            for j, (x,y) in enumerate(zip(Xtmp, Ytmp)):               
                gradient = gradients[j].to(device)
                # Compute influence
                est = - gradient.T@H_inv@(g_test-gradient)/N
                alphas[j] = losses[j] + np.array(est.cpu().detach())
                torch.cuda.empty_cache()

            pval = sum(alphas >= alphas[-1])/(N+1)            
            pvals_xtest[yhat], scores[yhat] = pval, list(alphas)
            print(pval)

        prediction_times.append(time.perf_counter() - start)
        pvals.append(pvals_xtest)
        
        if out_file:
            log_to_file(out_file, {"N": len(Xtrain),
                                   "prediction-times": prediction_times[-1],
                                   "p-values": pvals_xtest
                                  })
        
    return pvals, prediction_times

def ACP_O(model, Xtrain, Ytrain, Xtest, labels = [0,1], out_file = None, seed = 42, damp = 10**-3, batches = 1):
    """Runs ACP (ordinary) CP-IF to make a prediction for all points in Xtest
    """ 
    
    N = len(Xtrain)
    # Train model on D.
    model_D = model
    model_D = model_D.to(device)
    model_D.fit(Xtrain, Ytrain, seed = seed)
    torch.cuda.empty_cache() 
    # Estimate influence.
    gradients = []
    for x, y in zip(Xtrain, Ytrain):
        gradients.append(model_D.grad_z(x, y, flatten = True).cpu().detach())
        torch.cuda.empty_cache()
    gradients.append(None) #for the test point
    
    H_D = torch.zeros(model_D.count_params(), model_D.count_params()).to(device)
    Xtrain_splitted = np.array_split(Xtrain, batches)
    Ytrain_splitted = np.array_split(Ytrain, batches)
    for batch_X, batch_Y in zip(Xtrain_splitted, Ytrain_splitted):
        H_D += model_D.hessian_all_points(batch_X, batch_Y)
    H_D = H_D/batches
    H_D += torch.diag(Tensor([damp]*len(H_D))).to(device)
    torch.cuda.empty_cache()
    H_inv = torch.inverse(H_D)
    del H_D
    
    #Preliminary scores
    losses = [np.float64(model_D.compute_loss(x, y).cpu().detach()) for x, y in zip(Xtrain, Ytrain)]
    losses.append(None)
    
    pvals = []
    prediction_times = []
    
    for k, xtest in enumerate(Xtest):

        print("TEST: " +str(k+1))
        pvals_xtest = {}
        scores = {}
        start = time.perf_counter()
        
        for yhat in labels: 

            # Extended dataset
            Xtmp = np.row_stack((Xtrain, [xtest]))
            Ytmp = np.concatenate((Ytrain, [yhat]))
            alphas = np.zeros(len(Xtmp))
            # Obtain gradient on test point
            g_test = model_D.grad_z(Xtmp[-1,:], Ytmp[-1], flatten = True)
            # Obtain loss on test point
            loss_test = np.float64(model_D.compute_loss(Xtmp[-1,:], Ytmp[-1]).cpu().detach())
            gradients[-1] = g_test
            losses[-1] = loss_test
            
            for j, (x,y) in enumerate(zip(Xtmp, Ytmp)):                
                gradient = gradients[j].to(device)
                # Compute influence
                est = - gradient.T@H_inv@g_test/N
                alphas[j] = losses[j] + np.array(est.cpu().detach())
                torch.cuda.empty_cache()

            pval = sum(alphas >= alphas[-1])/(N+1)
            print(pval)
            pvals_xtest[yhat], scores[yhat] = pval, list(alphas)

        prediction_times.append(time.perf_counter() - start)
        pvals.append(pvals_xtest)
        
        if out_file:
            log_to_file(out_file, {"N": len(Xtrain),
                                   "prediction-times": prediction_times[-1],
                                   "p-values": pvals_xtest
                                  })
        
    return pvals, prediction_times