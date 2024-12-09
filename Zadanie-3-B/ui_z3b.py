# Import knižníc
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
            nn.Linear(8, 512),      # Vstupná vrstva -> zvýšenie na 512 neurónov
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),   # Skrytá vrstva 1 -> zvýšenie na 256 neurónov
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),   # Skrytá vrstva 2 -> zvýšenie na 128 neurónov
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 64),    # Skrytá vrstva 3 -> ponechanie
            nn.ReLU(),
            nn.Linear(64, 1)       # Výstupná vrstva
        )
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


# Funkcia na tréning s optimalizátormi
def train_with_optimizers(program_select, model_class, train_loader, test_loader, criterion, epochs):
    """optimizers = {
        "SGD": optim.SGD(model_class().parameters(), lr=0.001),
        "SGD with Momentum": optim.SGD(model_class().parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(model_class().parameters(), lr=0.01),
    }"""

    if program_select==1:
        optimizer={"SGD": optim.SGD(model_class().parameters(), lr=0.01)}
    elif program_select==2:
        optimizer={"SGD with Momentum": optim.SGD(model_class().parameters(), lr=0.01, momentum=0.9)}
    elif program_select==3:
        optimizer={"Adam": optim.Adam(model_class().parameters(), lr=0.001)}

    results = {}
    for name, optimizer in optimizer.items():
        print(f"\nTraining with {name} optimizer")
        model = model_class()
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.flatten(), targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss / len(train_loader))

            model.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for features, targets in test_loader:
                    outputs = model(features)
                    loss = criterion(outputs.flatten(), targets)
                    epoch_test_loss += loss.item()

            test_losses.append(epoch_test_loss / len(test_loader))
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        results[name] = {"train_losses": train_losses, "test_losses": test_losses}

    return results

# Grafické zobrazenie
def visual(results):
    for name, result in results.items():
        plt.plot(range(1, 21), result["train_losses"], label=f"{name} - Train")
        plt.plot(range(1, 21), result["test_losses"], label=f"{name} - Test")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Porovnanie optimalizátorov")
    plt.show()

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

while True:
    main_select=int(input("Zadajte ulohu na vypracovanie:\n1 - Poduloha\n2 - Backpropagation\nIne - Koniec\n>:"))
    print(f"{"-"*100}")
    if main_select==1 or main_select==2:
        program_select=int(input("Zadanie metody trenovania\n1 - SDG\n2 - SDG s momentom\n3 - ADAM\n>:"))
    else:
        print("Ukoncenie programu!"),exit()

    if program_select in (1,2,3):
        epochs=input("Pocet epochov (default = 20):") or 20
        epochs=int(epochs)

        batch_size=input("Batch size (default = 16):") or 16
        batch_size=int(batch_size)

        train_loader, test_loader=preparation(batch_size)
        #train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        #test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        criterion = nn.MSELoss()
        results = train_with_optimizers(program_select,HousingModel, train_loader, test_loader, criterion, epochs)
        visual(results)
        print(f"{"-"*100}")
    else:
        print("Ukoncenie programu!"),exit()

#criterion = nn.MSELoss()
#results = train_with_optimizers(HousingModel, train_loader, test_loader, criterion, epochs=20)
#visual(results)