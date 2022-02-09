from utils import *
from models import *
import copy

from third_party.RAPS.RAPS_conformal import *
from third_party.ARC.arc import models, methods, black_boxes, coverage

def deleted_full_CP(model, Xtrain, Ytrain, xtest, yhat, labels = [0,1], seed = 42):
    
    """Runs deleted full CP to make a prediction for (xtest, yhat).
    """
    N = len(Xtrain)
    Xtmp = np.row_stack((Xtrain, [xtest]))
    Ytmp = np.concatenate((Ytrain, [yhat]))
    alphas = np.zeros(len(Xtmp))

    for i, (x, y) in enumerate(zip(Xtmp, Ytmp)):
        
        new_model = copy.deepcopy(model)
        new_model = new_model.to(device)
        
        new_model.fit(np.delete(Xtmp, i, 0), np.delete(Ytmp, i), seed = seed)        
        alphas[i] = np.float64(new_model.compute_loss(x, y).cpu().detach())
        

    pval = sum(alphas >= alphas[-1])/(N+1)
    
    return pval, list(alphas)

def ordinary_full_CP(model, Xtrain, Ytrain, xtest, yhat, labels = [0,1], seed = 42):
    
    """Runs ordinary full CP to make a prediction for (xtest, yhat).
    """
    N = len(Xtrain)
    Xtmp = np.row_stack((Xtrain, [xtest]))
    Ytmp = np.concatenate((Ytrain, [yhat]))
    
    # Train model on full data.
    model.fit(Xtmp, Ytmp, seed = seed)
    alphas = [np.float64(model.compute_loss(xtmp, ytmp).cpu().detach()) for xtmp, ytmp in zip(Xtmp, Ytmp)]
    
    pval = sum(alphas >= alphas[-1])/(N+1)
    
    return pval, list(alphas)

def split_cp(model, Xtrain, Ytrain, Xtest, labels = [0,1], out_file = None, validation_split = 0.2, seed = 42):
    
    """Runs split/inductive CP for all points in Xtest
    """
    
    # Split data in training and calibration set
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size = validation_split, random_state = seed)
    # Train model on training set
    model.fit(Xtrain, Ytrain, seed = seed)
    
    #Calculate non-conformity scores in calibration set
    alphas = [np.float64(model.compute_loss(xval, yval).cpu().detach()) for xval, yval in zip(Xval, Yval)]
    pvals = []
    prediction_times = []
    
    for k, xtest in enumerate(Xtest):
        
        pvals_xtest = {}
        start = time.perf_counter()
        
        for yhat in labels:

            alpha_test = np.float64(model.compute_loss(xtest, yhat).cpu().detach())
            pval = (sum(alphas >= alpha_test)+1)/(len(Xval) + 1)
            pvals_xtest[yhat] = pval

        prediction_times.append(time.perf_counter() - start)
        pvals.append(pvals_xtest)
        
        if out_file:
            log_to_file(out_file, {"N": len(Xtrain),
                                   "prediction-times": prediction_times[-1],
                                   "p-values": pvals_xtest
                                  })
        
    return pvals, prediction_times

def RAPS(model, Xtrain, Ytrain, Xtest, Ytest, labels = [0,1], out_file = None, eps = None, seed = 42, save_results = True):
    
    """Runs RAPS for all points in Xtest (Angelopoulos et al.)
    """
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size = 0.2, random_state = seed)
    prediction_times = {}
    
    start = time.perf_counter()
    model.fit(Xtrain, Ytrain, seed = seed)
    calib_data = create_set(Xval, Yval)
    calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=128, shuffle=True, pin_memory=True)
    test_data = create_set(Xtest, Ytest)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, pin_memory=True)
    prediction_times["Training"] = [time.perf_counter() - start]
    
    if eps == None:
        epsilons = np.arange(0, 1.01,0.01)
    else:
        epsilons = [eps]
        
    sizes = {}
    sets = {}
    
    for i, epsilon in enumerate(epsilons):
        cmodel = ConformalModel(model, calib_loader, alpha=epsilon, lamda_criterion='size', allow_zero_sets=True)
        for j in range(len(Ytest)):
            
            start = time.perf_counter()
            _, set = cmodel(check_tensor(np.expand_dims(Xtest[j], axis=0)).to(device))

            if epsilon not in sets:
                sets[epsilon] = [list(set[0])]
                sizes[epsilon] = [len(list(set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(set[0]))
                sizes[epsilon].append(len(list(set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)
    
    if save_results:
        
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

def APS(model, Xtrain, Ytrain, Xtest, Ytest, labels = [0,1], seed = 42, out_file = None, eps = None, save_results = True):
    
    """Runs APS for all points in Xtest (Romano et al.)
    """
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size = 0.2, random_state = seed)
    prediction_times = {}
    
    start = time.perf_counter()
    model.fit(Xtrain, Ytrain, seed = seed)
    prediction_times["Training"] = [time.perf_counter() - start]
    
    if eps == None:
        epsilons = np.arange(0, 1.01,0.01)
    else:
        epsilons = [eps]
        
    sizes = {}
    sets = {}
       
    for i, epsilon in enumerate(epsilons):
        method_sc = methods.SplitConformal(Xval, Yval, model, alpha = epsilon, seed = seed, verbose = True)
        for j in range(len(Ytest)):            
            start = time.perf_counter()
            set = method_sc.predict(check_tensor(np.expand_dims(Xtest[j], axis=0)).to(device))

            if epsilon not in sets:
                sets[epsilon] = [list(set[0])]
                sizes[epsilon] = [len(list(set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(set[0]))
                sizes[epsilon].append(len(list(set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)
    
    if save_results:
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

def CV_plus(model, Xtrain, Ytrain, Xtest, Ytest, labels = [0,1], seed = 42, out_file = None, eps = None, save_results = True):
    
    """Runs CV+ for all points in Xtest (Romano et al.)
    """
    
    prediction_times = {}
    
    start = time.perf_counter()
    method_cv = methods.CVPlus(Xtrain, Ytrain, model, n_folds=5, seed = seed, verbose = False)
    prediction_times["Training"] = [time.perf_counter() - start]
    
    if eps == None:
        epsilons = np.arange(0, 1.01,0.01)
    else:
        epsilons = [eps]
        
    sizes = {}
    sets = {}
       
    for i, epsilon in enumerate(epsilons):
        for j in range(len(Ytest)):
            
            start = time.perf_counter()
            set = method_cv.predict(check_tensor(np.expand_dims(Xtest[j], axis=0)).to(device), alpha = epsilon)

            if epsilon not in sets:
                sets[epsilon] = [list(set[0])]
                sizes[epsilon] = [len(list(set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(set[0]))
                sizes[epsilon].append(len(list(set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)
    
    if save_results:
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

def JK_plus(model, Xtrain, Ytrain, Xtest, Ytest, labels = [0,1], seed = 42, out_file = None, eps = None, save_results = True):
    
    """Runs JK+ for all points in Xtest (Romano et al.)
    """
    prediction_times = {}
    
    start = time.perf_counter()
    method_jk = methods.JackknifePlus(Xtrain, Ytrain, model, seed = seed, verbose = True)
    prediction_times["Training"] = [time.perf_counter() - start]
    
    if eps == None:
        epsilons = np.arange(0, 1.01,0.01)
    else:
        epsilons = [eps]
        
    sizes = {}
    sets = {}
       
    for i, epsilon in enumerate(epsilons):
        for j in range(len(Ytest)):
            
            start = time.perf_counter()
            set = method_jk.predict(check_tensor(np.expand_dims(Xtest[j], axis=0)).to(device), alpha = epsilon)

            if epsilon not in sets:
                sets[epsilon] = [list(set[0])]
                sizes[epsilon] = [len(list(set[0]))]
                prediction_times[epsilon] = [time.perf_counter() - start]
            else:
                sets[epsilon].append(list(set[0]))
                sizes[epsilon].append(len(list(set[0])))
                prediction_times[epsilon].append(time.perf_counter() - start)
    
    if save_results:
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

  


