# Import knižníc
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 1. Načítanie a spracovanie datasetu
#print("Priprava dat")
def preparation(batch_size):
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    train_dataset = CaliforniaHousingDataset(X_train, y_train)
    test_dataset = CaliforniaHousingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader,test_loader

# Dataset trieda
class CaliforniaHousingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        #self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


# Funkcia na tréning s optimalizátormi
def train_with_optimizers(opt_name, optimizer, model, train_loader, test_loader, crit, epochs):
    print(f"\nZvoleny {opt_name} optimizator")
    #model = model_class()
    train_losses = []
    test_losses = []
    results = {}

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = crit(outputs.flatten(), targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)
                loss = crit(outputs.flatten(), targets)
                epoch_test_loss += loss.item()
        test_losses.append(epoch_test_loss / len(test_loader))
        print(f"Epoch c. {epoch+1}, TrainL: {train_losses[-1]:.4f}, TestL: {test_losses[-1]:.4f}")

    results[opt_name] = {"train_losses": train_losses, "test_losses": test_losses}

    return results

# Grafické zobrazenie
def visual(results, epochs):
    for name, result in results.items():
        plt.plot(range(1, epochs+1), result["train_losses"], label=f"{name} - Train", color='Blue')
        plt.plot(range(1, epochs+1), result["test_losses"], label=f"{name} - Test", color='Purple')
    

    plt.xlabel("Epochy")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Porovnanie optimalizátorov")
    plt.show()

while True:
    print(f"{"-"*100}")
    program_select=int(input("Zadajte metodu trenovania\n1 - SDG\n2 - SDG s momentom\n3 - ADAM\n>:"))

    epochs=input("Pocet epochov (default = 40):") or 40
    epochs=int(epochs)

    batch_size=input("Batch size (default = 32):") or 32
    batch_size=int(batch_size)
    print(f"{"-"*100}")

    crit = nn.MSELoss()
    train_loader, test_loader=preparation(batch_size)

    if program_select==1:
        opt_name="SGD"
        model=HousingModel()
        optimizer=optim.SGD(model.parameters(), lr=0.01)

    elif program_select==2:
        opt_name="SGD s momentom"
        model=HousingModel()
        optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    elif program_select==3:
        opt_name="ADAM"
        model=HousingModel()
        optimizer= optim.Adam(model.parameters(), lr=0.001)
    
    else:
        print("Ukoncenie programu"),exit()

    results= train_with_optimizers(opt_name, optimizer, model, train_loader, test_loader, crit, epochs)
    visual(results, epochs)
    print(f"{"-"*100}")