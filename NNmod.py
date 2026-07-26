import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

class MomMortMod(nn.Module):
    def __init__(self, input_size, hidden_layers, drop_prob=0.2):
        super().__init__()
        self.mid_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        current_dim = input_size
        
        for hidden_dim in hidden_layers:
            self.mid_layers.append(nn.Linear(current_dim, hidden_dim))
            self.drop_layers.append(nn.Dropout(p=drop_prob))
            current_dim = hidden_dim
        
        self.output_layer = nn.Linear(current_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer, drop in zip(self.mid_layers, self.drop_layers):
            x = self.relu(layer(x))
            x = drop(x)
        x = self.output_layer(x)
        return self.sigmoid(x)

class NN_PredMod:
    def __init__(self, input_size, hidden_layers, epochs=100, lr=0.001):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.lr = lr                                
        self.epochs = epochs
        
        self.criterion = nn.BCELoss()
        self.model = None
        self.optimizer = None

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float().view(-1, 1)
        
        self.model = MomMortMod(self.input_size, self.hidden_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            pred_train = self.model(X_train)
            loss = self.criterion(pred_train, y_train)
            loss.backward()
            self.optimizer.step()
            
            # if (epoch+1) % 25 == 0:
            #     print(f"Epoch: {epoch + 1:3d} | Train Loss: {loss.item():.4f}")
    
    def predict(self, X_test):
        X_test = torch.from_numpy(X_test).float()
        self.model.eval()
        with torch.no_grad():
            pred_test = self.model(X_test)
        return pred_test.numpy().flatten()
    
    def get_loss(self, y_hat, y):
        y_hat = torch.from_numpy(y_hat).float()
        y = torch.from_numpy(y).float()
        self.model.eval()
        with torch.no_grad():
            loss = self.criterion(y_hat, y)
        return loss.item()

class NN_CVmod:
    def __init__(self, input_size, hidden_layers, epochs=100, lr=0.001):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.lr = lr                                
        self.epochs = epochs
    
    def NN_CV(self, attributes, response, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mod_class = NN_PredMod(self.input_size, self.hidden_layers)
        auc_log = []
        train_log = []
        valid_log = []
        
        for fold, (train_ids, val_ids) in enumerate(kf.split(attributes)):
            print(f"Fold: {fold+1} of {k}")
            
            X_train = attributes[train_ids,:]
            X_val = attributes[val_ids,:]
            y_train = response[train_ids]
            y_val = response[val_ids]
            
            mod_class.fit(X_train, y_train)
            train_preds = mod_class.predict(X_train)
            val_preds = mod_class.predict(X_val)
            precision, recall, thresholds = precision_recall_curve(y_val, val_preds)
            PrRe_auc = auc(recall, precision)
            auc_log.append(PrRe_auc)
            train_loss = mod_class.get_loss(train_preds, y_train)
            valid_loss = mod_class.get_loss(val_preds, y_val)
            train_log.append(train_loss)
            valid_log.append(valid_loss)
            
            
        return np.mean(train_log), np.mean(valid_log), np.mean(auc_log)


            
            