from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Trieda pre dataset, ktorý prispôsobuje dáta
class CaliHousingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    # Počet vzoriek v datasete
    def __len__(self):
        return len(self.features)

    # Získanie jednej vzorky (vstup a cieľ)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Definícia modelu neurónovej siete
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    # Dopredný prechod cez sieť
    def forward(self, x):
        return self.model(x)

    # Nepoužívaná funkcia...
    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


# Funkcia na prípravu dát
def preparation(batch_size):
    # Nacitaj data
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Rozdelenie dát podľa zadania (80% trening, 20%v testovanie)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizácia dát
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Vytvorenie datasetov a DataLoaderov
    train_dataset = CaliHousingDataset(X_train, y_train)
    test_dataset = CaliHousingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

# Funkcia na tréning modelu
def train_with_optimizers(opt_name, optimizer, model, train_loader, test_loader, crit, epochs):
    print(f"Zvoleny {opt_name} optimizator")
    train_losses = []
    test_losses = []
    results = {}

    for epoch in range(epochs):
        # Nastavenie modelu do tréningového režimu, aktualizácia váh
        model.train()
        # Resetovanie straty
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = crit(outputs.flatten(), targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Priemerná strata za epochu
        train_losses.append(epoch_train_loss / len(train_loader))

        # Prepnutie do eval režimu (bez aktualizácie váh)
        model.eval()
        # Resetovanie straty
        epoch_test_loss = 0.0
        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)
                loss = crit(outputs.flatten(), targets)
                epoch_test_loss += loss.item()
        test_losses.append(epoch_test_loss / len(test_loader))
        print(f"Epoch c. {epoch+1} => Train loss: {train_losses[-1]:.4f}, Test loss: {test_losses[-1]:.4f}")

    results[opt_name] = {"train_losses": train_losses, "test_losses": test_losses}
    return results

# Funkcia na vizualizáciu priebehu tréningu
def visual(results, epochs, name):
    train_losses = results[name]["train_losses"]
    test_losses = results[name]["test_losses"]

    # Kreslenie grafov
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color='Purple')
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color='Blue')
    
    plt.xlabel("Epochy")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.title(f"Výsledky tréningu - {name}")
    plt.show()


# Hlavná slučka
while True:
    print(f"{"-"*100}")
    program_select=int(input("Zadajte metodu trenovania\n1 - SDG\n2 - SDG s momentom\n3 - ADAM\n>:"))

    # Aby sa zbytocne nepytal na batch a epochy, porieseny aj nespravny vystup
    if program_select not in (1,2,3):
        print("Ukoncenie programu (chybny vstup)!"),exit()

    # Input poctu epochov
    epochs=input("Pocet epochov (default = 20):") or 20
    epochs=int(epochs)

    # Input velkosti batch
    batch_size=input("Batch size (default = 16):") or 16
    batch_size=int(batch_size)
    print(f"{"-"*100}")

    # Stratová funkcia
    crit = nn.MSELoss()
    train_loader, test_loader=preparation(batch_size)

    if program_select==1:
        opt_name="SGD"
        model=HousingModel()
        optimizer=optim.SGD(model.parameters(), lr=0.01)

    elif program_select==2:
        opt_name="SGD s momentom"
        model=HousingModel()
        optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif program_select==3:
        opt_name="ADAM"
        model=HousingModel()
        optimizer= optim.Adam(model.parameters(), lr=0.001)

    # Spustenie tréningu a vizualizácia výsledkov
    results= train_with_optimizers(opt_name, optimizer, model, train_loader, test_loader, crit, epochs)
    visual(results, epochs, opt_name)