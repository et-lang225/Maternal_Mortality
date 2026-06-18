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

    def fit(self, X, y):
        X_ten = torch.from_numpy(X).float()
        y_ten = torch.from_numpy(y).float().unsqueeze(1)
        loss_log = []
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            Mort_hat = self.model(X_ten)
            loss = self.criterion(Mort_hat, y_ten)
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            loss_log.append(loss.item())
            if epoch+1 > 5:
                if loss_log[-1]/np.linalg.norm(loss_log[(epoch-5):]) < 1e-3:
                    print("Converged!")
                    break
            