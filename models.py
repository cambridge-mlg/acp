import torch
from torch import nn
from torch import Tensor, flatten
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import Flatten , Dropout, BatchNorm1d, ReLU, Conv2d, Linear, MaxPool2d, BCELoss, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.nn.init import kaiming_uniform_, xavier_uniform_ 
from torch.autograd import grad
from utils import *
from methods import *

device = torch.device('cuda')
print(f'Using {device} device')

class NeuralNetworkTemplate(nn.Module):
    
    def __init__(self, input_size: int, out_size: int, l2_reg: float = 0, seed: int = 42):        
        super(NeuralNetworkTemplate, self).__init__()
        
        self.seed = seed
        self.input_size = input_size
        self.out_size = out_size
        self.l2_reg = l2_reg
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
    def forward(self, x):  
        "Forward method"
        return 
    
    def training_step(self, batch, validation = False):
        features, targets = batch
        outputs = self(features)
        loss = self.loss(targets, outputs, self.l2_reg)  
        if validation == True:
            return {'val_loss': loss}
        else:
            return loss
        
    def validation_epoch_end(self, outputs):          
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch: {} - Validation Loss: {:.4f}".format(epoch+1, result))
        
    def fit(self, x, y, epochs = 200, learning_rate = 0.001, validation_split = 0.2, batch_size = 100, early_stopping = True, 
            patience = 5, n_epochs_min = 10, verbose = 1, n_iter_print = 10, seed = 42, num_workers = 0): 
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
        x = check_tensor(x).to(device)
        y = check_tensor(y).to(device)            
        x, y, x_val, y_val = make_val_split(x, y, val_split_prop=validation_split, seed=seed)
        train_data = create_set(x, y)
        val_data = create_set(x_val, y_val)
        train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_data, batch_size, num_workers=num_workers, pin_memory=False)
       
        history = []
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        best_loss = np.inf
        p_curr = 0

        for epoch in range(epochs):
            # Training 
            for batch in train_loader:
                self.train()
                loss = self.training_step(batch)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # Validation
            current_loss = self._evaluate(val_loader)['val_loss']
            if (verbose != 0 and ((epoch+1) % n_iter_print == 0) or epoch==0):
                self.epoch_end(epoch, current_loss)
            history.append(current_loss)
            
            #Early stopping and patience
            if early_stopping and ((epoch+1)>n_epochs_min):
                if current_loss < best_loss:
                    best_loss = current_loss
                    p_curr = 0
                else:
                    p_curr = p_curr + 1
                    
                if p_curr > patience:
                    return         
        return
    
    def loss(self, targets, outputs, l2_reg = 0):
        "Method for computing the loss in backprop"
        return
    
    def _evaluate(self, val_loader):
        outputs = [self.training_step(batch, validation = True) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def predict_proba(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            self.eval()
            outputs = self(x)
            if x.dim() == 1:
                outputs = outputs.unsqueeze(0)
            outputs = F.softmax(outputs, dim = 1) 
        return outputs
        
    def predict(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            self.eval()
            outputs = self(x)  
            if x.dim() == 1:
                outputs = outputs.unsqueeze(0)
            outputs = F.softmax(outputs, dim = 1)
            predictions = []
            for i in range(len(outputs)):
                class_pred = torch.argmax(outputs[i])
                predictions.append(class_pred)
        return check_tensor(predictions)
    
    def compute_loss(self, x, y, gpu = 0):
        self.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(y).unsqueeze(0)
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self(x)
        return self.loss(targets, outputs, self.l2_reg)

    def grad_z(self, x, targets, gpu = 0, flatten = False):
        """Returns:
        grad_z: list of torch tensor, containing the gradients from model parameters to loss"""
        self.zero_grad()
        x = check_tensor(x).unsqueeze(0)
        targets = check_tensor(targets).unsqueeze(0)
        # initialize
        if gpu >= 0:
            x, targets = x.to(device), targets.to(device)
        outputs = self(x)
        loss = self.loss(targets, outputs, self.l2_reg)
        # Compute sum of gradients from model parameters to loss
        params = [ p for p in self.parameters() if p.requires_grad ]
        gradient = list(grad(loss, params, create_graph=True))
        if flatten:
            return flatten_gradient(gradient)           
        return gradient
    
    def hessian_all_points(self, Xtrain, Ytrain, gpu = 0):
        
        def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):

            if torch.is_tensor(inputs):
                inputs = [inputs]
            else:
                inputs = list(inputs)
            grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                        allow_unused=True,
                                        retain_graph=retain_graph,
                                        create_graph=create_graph)
            grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
            return torch.cat([x.contiguous().view(-1) for x in grads])
        
        Xtrain = check_tensor(Xtrain)
        Ytrain = check_tensor(Ytrain)
        
        if gpu >= 0:
            Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
            
        outputs = self(Xtrain)        
        loss = self.loss(Ytrain, outputs, self.l2_reg)        
        inputs = self.parameters()
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
        return sum(p.numel() for p in self.parameters())

class NeuralNetwork(NeuralNetworkTemplate):
        
    def __init__(self, input_size: int, num_neurons: list, out_size: int, l2_reg: float = 0, seed: int = 42):
        super().__init__(input_size = input_size, out_size = out_size, l2_reg = l2_reg, seed = seed) 
        
        self.num_neurons = num_neurons

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
        self.layers = nn.ModuleList()
        self.layers.append(Linear(input_size, num_neurons[0]))
        kaiming_uniform_(self.layers[-1].weight, nonlinearity='relu')
        for i in range(len(num_neurons)-1):
            self.layers.append(Linear(num_neurons[i], num_neurons[i+1]))
            kaiming_uniform_(self.layers[-1].weight, nonlinearity='relu')                      
        self.layers.append(Linear(num_neurons[-1], out_size))
        xavier_uniform_(self.layers[-1].weight)
        
    def forward(self, x):
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i!=(len(self.layers)-1): #dont include ReLU in output layer
                x = F.relu(x)               
        return x        
        
    def loss(self, targets, outputs, l2_reg = 0):
        if self.out_size == 1:
            targets = targets.unsqueeze(1)
            loss_func = BCEWithLogitsLoss(reduction = "mean") #includes sigmoid act.
        else:
            targets = targets.long()
            loss_func = CrossEntropyLoss(reduction = "mean") #includes softmax act.
        loss = loss_func(outputs, targets)
            
        loss = loss + l2_regularization(self.layers, self.l2_reg)
        
        return loss
    
    def predict_proba(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            self.eval()
            outputs = self(x)
            if self.out_size != 1:
                if x.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                outputs = F.softmax(outputs, dim = 1)
            else:
                outputs = torch.sigmoid(outputs) 
        return outputs
        
    def predict(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            self.eval()
            outputs = self(x)
            if self.out_size != 1:
                if x.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                outputs = F.softmax(outputs, dim = 1)
            else:
                outputs = torch.sigmoid(outputs) 
            predictions = []
            for i in range(len(outputs)):
                if self.out_size == 1:
                    class_pred = 1 if outputs[i] >= 0.5 else 0 #threshold = 0.5
                else:
                    class_pred = torch.argmax(outputs[i])
                predictions.append(class_pred)
        return check_tensor(predictions)
    
class ConvolutionalNeuralNetwork(NeuralNetworkTemplate):
        
    def __init__(self, channels: int, out_size: int, l2_reg: float = 0, seed: int = 42):
        
        super().__init__(input_size = None, out_size = out_size, l2_reg = l2_reg, seed = seed)
        
        self.channels = channels

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        self.conv1 = Conv2d(in_channels=channels, out_channels=16, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(4, 4), stride=(2, 2))
        
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(4, 4), stride=(2, 2))
        
        if channels == 3: #CIFAR-10
            fc1_in_size = 288
        elif channels == 1: #MNIST
            fc1_in_size = 128
        
        self.fc1 = Linear(in_features=fc1_in_size, out_features=out_size) #changes with MNIST, this is for CIFAR
        
        
    def forward(self, x):       
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        return x
              
    def loss(self, targets, outputs, l2_reg = 0):
        targets = targets.long()
        loss_func = CrossEntropyLoss(reduction = "mean") #includes softmax act.
        loss = loss_func(outputs, targets)      
        return loss
    
class LogisticRegression(NeuralNetworkTemplate):
        
    def __init__(self, input_size: int, out_size: int, l2_reg: float = 0, seed: int = 42):
        
        super().__init__(input_size = input_size, out_size = out_size, l2_reg = l2_reg, seed = seed) 

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
        self.layers = nn.ModuleList()
        self.layers.append(Linear(input_size, out_size))
        
    def forward(self, x):
        outputs = self.layers[0](x)
        return outputs
    
    def loss(self, targets, outputs, l2_reg = 0):
        if self.out_size == 1:
            targets = targets.unsqueeze(1)
            loss_func = BCEWithLogitsLoss(reduction = "mean") #includes sigmoid act.
        else:
            targets = targets.long()
            loss_func = CrossEntropyLoss(reduction = "mean") #includes softmax act.
        loss = loss_func(outputs, targets)            
        loss = loss + l2_regularization(self.layers, self.l2_reg)
        
        return loss
    
    def predict_proba(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            sef.eval()
            outputs = self(x)
            if self.out_size != 1:
                if x.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                outputs = F.softmax(outputs, dim = 1)
            else:
                outputs = torch.sigmoid(outputs) 
        return outputs
        
    def predict(self, x):
        x = check_tensor(x).to(device)
        # Disable grad
        with torch.no_grad():
            self.eval()
            outputs = self(x)
            if self.out_size != 1:
                if x.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                outputs = F.softmax(outputs, dim = 1)
            else:
                outputs = torch.sigmoid(outputs) 
            predictions = []
            for i in range(len(outputs)):
                if self.out_size == 1:
                    class_pred = 1 if outputs[i] >= 0.5 else 0 #threshold = 0.5
                else:
                    class_pred = torch.argmax(outputs[i])
                predictions.append(class_pred)
        return check_tensor(predictions)
    
    def closed_form_hessian(self, X):
        preds = self.predict_proba(X).cpu().detach().numpy()
        features = len(X[0])
        classes = self.out_size
        hessian = np.zeros((classes,classes,features,features)) 
        for i in range(len(X)):
            hessian += hessian_multi_LR(X[i].reshape(1,-1), preds[i].reshape(1,-1), classes)
        return Tensor(hessian).to(device)
        
    def hessian_multi_LR(X, Y, classes):
        features = len(X[0])
        I = np.identity(classes)
        N = len(X)
        H = np.zeros((classes,classes,features,features))  
        for i in range(classes):
            for j in range(classes):
                aux = np.repeat(np.multiply(Y[:,i], (I[i,j]-Y[:,j])) , features, axis=0)
                aux = aux.reshape(features, N)
                H[i,j] = np.matmul(np.multiply(aux, X.T), X) + np.diag([10**-10]*features)
        return H
    
    
class AE(nn.Module):
    
    def __init__(self, input_shape, embedding_size, seed = 42):
        super().__init__()
        
        self.seed = seed
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
        
        self.encoder = nn.Sequential(
            Linear(input_shape, 128),
            ReLU(True),
            Dropout(0.2),
            Linear(128, 64),
            ReLU(True),
            Dropout(0.2),
            Linear(64, embedding_size),
        )
        
        self.decoder = nn.Sequential(
            BatchNorm1d(embedding_size),
            Linear(embedding_size, 64),
            ReLU(True),
            Dropout(0.2),
            Linear(64, 128),
            ReLU(True),
            Dropout(0.2),
            Linear(128, input_shape)
        )

    def forward(self, x):       
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, validation = False):
        features = batch
        outputs = self(features)
        criterion = MSELoss()
        loss = criterion(outputs, features)
        if validation == True:
            return {'val_loss': loss}
        else:
            return loss      
    
    def validation_epoch_end(self, outputs):          
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch: {} - Validation Loss: {:.4f}".format(epoch+1, result))
        
    def fit(self, x, epochs, learning_rate, validation_split = 0.2, batch_size = 100, early_stopping = True, 
            patience = 5, n_epochs_min = 10, verbose = 1, n_iter_print = 10, seed = 42): 
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        x = check_tensor(x).to(device)            
        x, x_val = train_test_split(x, test_size=validation_split, random_state=seed, shuffle=True)
        train_loader = DataLoader(x, batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(x_val, batch_size, num_workers=0, pin_memory=False)
       
        history = []
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        best_loss = np.inf
        p_curr = 0

        for epoch in range(epochs):
            # Training 
            for batch in train_loader:
                self.train()
                loss = self.training_step(batch)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # Validation
            current_loss = self._evaluate(val_loader)['val_loss']
            if (verbose != 0 and ((epoch+1) % n_iter_print == 0) or epoch==0):
                self.epoch_end(epoch, current_loss)
            history.append(current_loss)
            
            #Early stopping and patience
            if early_stopping and ((epoch+1)>n_epochs_min):
                if current_loss < best_loss:
                    best_loss = current_loss
                    p_curr = 0
                else:
                    p_curr = p_curr + 1
                    
                if p_curr > patience:
                    return         
        return
    
    def _evaluate(self, val_loader):
        outputs = [self.training_step(batch, validation = True) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def embed(self,x):
        x = check_tensor(x).to(device) 
        with torch.no_grad():
            self.eval()
            embedding = self.encoder(x).cpu().detach()
        return embedding

    
                
        
