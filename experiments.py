from utils import *
from methods import *
from others import *
from models import *
import argparse
import tensorflow
from tensorflow.keras.datasets import mnist, cifar10
from folktables import ACSDataSource, ACSIncome

def experiments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("function", help="CP function to run (full_CP, ACP_D, ordinary_full_CP, ACP_O, SCP, RAPS, APS, CV_plus, JK_plus)")
    parser.add_argument("dataset", help="dataset (synthetic, MNIST, US_Census, CIFAR-10)")
    parser.add_argument("model", help="Neural Network A, B, C, LR or CNN")
    parser.add_argument("--reg", help="value l2 regularization term", type = float, default = 0.00001)
    parser.add_argument("--seed", help="initial seed", type = int, default = 1000)
    parser.add_argument("--test", help="test set size", type = int, default = 100)
    parser.add_argument("--dir", help="output dir", default = "results/")
    parser.add_argument("--embedding_size", help="embedding size for the autoencoder", type = int, default = 8)
    parser.add_argument("--validation_split", help="split for calibration set in SCP", type = float, default = 0.2)
    parser.add_argument("--epsilon", help="value of epsilon for RAPS, APS, JK+ or CV+", type = float, default = None)
    args = parser.parse_args()
    
    LOGS = args.dir
    N_TEST = args.test
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Loading data
    if args.dataset == "synthetic":
        Ns = np.logspace(1, 5, 13, dtype='int').tolist()[:10] 
        N_CLASSES = 5
        P = 10
        X, Y = make_classification(max(Ns)+N_TEST, P, n_classes=N_CLASSES, n_clusters_per_class = 1, n_informative = 3)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=N_TEST, random_state = args.seed)
    
    elif args.dataset in ["MNIST", "CIFAR-10"]:
        if args.dataset == "MNIST":
            (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data() #Load MNIST
        else:
            (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data() #Load CIFAR-10
          
        #Preprocessing data (2 scenarios: CNN and not-CNN (i.e. NN and LR))
        if args.model == "CNN":
            Xtrain = Xtrain.reshape(Xtrain.shape[0], -1, Xtrain.shape[1], Xtrain.shape[2])
            Xtest = Xtest.reshape(Xtest.shape[0], -1, Xtest.shape[1], Xtest.shape[2])
        else:
            Xtrain = normalize_flatten(Xtrain)
            Xtest = normalize_flatten(Xtest)
            autoencoder = AE(input_shape=Xtrain.shape[1], embedding_size=args.embedding_size, seed = args.seed).to(device)
            autoencoder.load_state_dict(torch.load("models/AE_"+args.dataset+"_"+str(args.embedding_size)))
            Xtrain = np.array(autoencoder.embed(Xtrain))
            Xtest = np.array(autoencoder.embed(Xtest))
            N, P = Xtrain.shape
        Ytrain = scalar(Ytrain)
        Ytest = scalar(Ytest)
        Xtest, Ytest = Xtest[:N_TEST], Ytest[:N_TEST]
        Ns = [Xtrain.shape[0]]
        
    elif args.dataset == "US_Census":
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=["NY"], download=True)
        features, label, group = ACSIncome.df_to_numpy(acs_data) #Load income data from US Census New York
        Xtrain, Xtest, Ytrain, Ytest, _, _ = train_test_split(features, label, group, test_size=0.1, random_state=args.seed)
        Xtest, Ytest = Xtest[:N_TEST], Ytest[:N_TEST]
        Ytrain = np.array([1 if x  else 0 for x in Ytrain])
        Ytest = np.array([1 if x  else 0 for x in Ytest])
        Ns = [Xtrain.shape[0]]
        N, P = Xtrain.shape
    else:
        print("Dataset not included")
        raise NotImplementedError
    
    LABELS = np.unique(Ytrain).tolist()
    out_size = len(LABELS)
    
    #Folder+file for saving results 
    logfile = LOGS + args.function + "_" + args.dataset + "_" + str(args.embedding_size) + "_" + str(args.model) + "_" + str(args.seed)

    for N in tqdm(Ns):
        
        #Invoking model at each iteration
        if args.model == "A":
            model = NeuralNetwork(input_size = P, num_neurons = [20,10], out_size = out_size, l2_reg = args.reg, seed = args.seed)
        elif args.model == "B":
            model = NeuralNetwork(input_size = P, num_neurons = [100], out_size = out_size, l2_reg = args.reg, seed = args.seed)
        elif args.model == "C":
            model = NeuralNetwork(input_size = P, num_neurons = [100, 50, 20], out_size = out_size, l2_reg = args.reg, seed = args.seed)
        elif args.model == "LR":
            model = LogisticRegression(input_size = P, out_size = out_size, l2_reg = args.reg, seed = args.seed)
        elif args.model == "CNN":
            model = ConvolutionalNeuralNetwork(channels = Xtrain.shape[1], out_size = out_size, l2_reg = args.reg, seed = args.seed)
        else:
            print("Model not implemented")
            raise NotImplementedError
       
        model = model.to(device)
    
        #Selecting and executing appropriate conformal predictor
        if args.function == "full_CP":
            _, _ = evaluate_cp_method(model, deleted_full_CP, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS,
                                      out_file=logfile, seed = args.seed) 
            
        elif args.function == "ordinary_full_CP":
            _, _ = evaluate_cp_method(model, ordinary_full_CP, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS,
                                      out_file=logfile, seed = args.seed)
            
        elif args.function == "ACP_D":
            _, _ = ACP_D(model, Xtrain[:N,:], Ytrain[:N], Xtest, labels=LABELS, out_file=logfile, seed = args.seed) 
            
        elif args.function == "ACP_O":
            _, _ = ACP_O(model, Xtrain[:N,:], Ytrain[:N], Xtest, labels=LABELS,out_file=logfile, seed = args.seed)
            
        elif args.function == "SCP":
            _, _ = split_cp(model, Xtrain[:N,:], Ytrain[:N], Xtest, labels=LABELS, out_file=logfile, 
                            validation_split = args.validation_split)     
            
        elif args.function == "RAPS":
            RAPS(model, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS, out_file=logfile, eps = args.epsilon, seed = args.seed)

        elif args.function == "APS":
            APS(model, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS, out_file=logfile, eps = args.epsilon, seed = args.seed)

        elif args.function == "CV_plus":
            CV_plus(model, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS, out_file=logfile, eps = args.epsilon, seed = args.seed)
            
        elif args.function == "JK_plus":
            JK_plus(model, Xtrain[:N,:], Ytrain[:N], Xtest, Ytest, labels=LABELS, out_file=logfile, eps = args.epsilon, seed = args.seed)

        else:
            print("CP function not implemented")
            raise NotImplementedError   

if __name__ == '__main__':
    experiments()
