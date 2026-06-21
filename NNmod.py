import torch
import torch.nn as nn
import numpy as np

class MomMortMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class NN_mod:
    def __init__(self, epochs=100, lr=0.01):
        self.model = MomMortMod()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)                                   
        self.epochs = epochs

    def fit(self, X_train, y_train, X_valid, y_valid):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float().unsqueeze(1)
        X_valid = torch.from_numpy(X_valid).float()
        y_valid = torch.from_numpy(y_valid).float().unsqueeze(1)
        train_log = []
        valid_log = []
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            pred_train = self.model(X_train)
            loss = self.criterion(pred_train, y_train)
            pred_valid = self.model(X_valid)
            valid_loss = self.criterion(pred_valid, y_valid)
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            train_log.append(loss.item())
            valid_log.append(valid_loss.item())
            
            