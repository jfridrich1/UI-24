import numpy as np
import matplotlib.pyplot as plt

# Základná trieda pre všetky moduly neurónovej siete
class Module:
    # Dopredný smer
    def forward(self, input):
        raise NotImplementedError("Forward musí byť implementovaný.")

    # Spätný smer
    def backward(self, gradient):
        raise NotImplementedError("Backward musí byť implementovaný.")
    
    # Aktualizácia váh
    def updateweights(this, learning_rate, momentum=0):
        pass

# Trieda reprezentujúca neurónovú sieť
class Model:
    # Zoznam modulov (vrstiev)
    def __init__(self, modules):
        self.modules = modules

    # Pridá novú vrstvu do modelu - nepouziva sa, pre debugovanie
    def add_module(self, module):
        self.modules.append(module)
    
    # Dopredný prechod cez všetky vrstvy modelu
    def forward(self, input):
        for module in self.modules:
            input = module.forward(input)
        return input
    
    # Spätný prechod cez všetky vrstvy modelu
    def backward(self, gradient):
        for module in reversed(self.modules):
            gradient = module.backward(gradient)

    # Aktualizácia váh vo všetkých vrstvách
    def updateweights(this, learning_rate, momentum=0):
        for module in this.modules:
            if isinstance(module, Linear):
                module.updateweights(learning_rate, momentum)


class Linear(Module):
    def __init__(self, inputsize, outputsize, activation="relu"):
        # Inicializácia váh a biasov
        if activation == "tanh":    # Xavier inicializácia pre tanh
            self.weight = np.random.randn(inputsize, outputsize) * np.sqrt(1 / inputsize)
        elif activation == "relu":  # He inicializácia pre ReLU
            self.weight = np.random.randn(inputsize, outputsize) * np.sqrt(2 / inputsize)
        else:
            self.weight = np.random.randn(inputsize, outputsize) * 0.01

        self.bias = np.zeros((1, outputsize))
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self.velocity_weight = np.zeros_like(self.weight)
        self.velocity_bias = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input          # Uchovanie vstupu pre spätný prechod
        return np.dot(input, self.weight) + self.bias

    def backward(self, gradient):
        self.grad_weight = np.dot(self.input.T, gradient)
        self.grad_bias = np.sum(gradient, axis=0, keepdims=True)
        return np.dot(gradient, self.weight.T)  # Gradient propagovaný späť

    # Aktualizácia váh a biasov pomocou gradient descent a momenta
    def updateweights(self, learning_rate, momentum=0):
        self.velocity_weight = momentum * self.velocity_weight - learning_rate * self.grad_weight
        self.velocity_bias = momentum * self.velocity_bias - learning_rate * self.grad_bias
        self.weight += self.velocity_weight
        self.bias += self.velocity_bias


    
class Sigmoid(Module):
    def __init__(self):
        # Uchováva výstup dopredného smeru pre deriváciu
        self.output = None
    
    def forward(self, input):
        # Výpočet sigmoid funkcie
        self.output = 1 / (1 + np.exp(-input))  # Výpočet sigmoid
        return self.output

    def backward(self, gradient):
        # Derivácia sigmoid
        sigmoid_derivative = self.output * (1 - self.output)
        return gradient * sigmoid_derivative


class Tanh(Module):
    def __init__(self):
        self.output = None  # Uchová výstup z dopredného smeru

    def forward(self, input):
        # Výpočet tanh funkcie
        self.output = np.tanh(input)
        return self.output

    def backward(self, gradient):
        # Derivácia tanh
        tanh_derivative = 1 - self.output ** 2
        return gradient * tanh_derivative


class ReLU(Module):
    def __init__(self):
        self.input = None  # Uchová vstup z dopredného smeru

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)  # Výpočet ReLU

    def backward(self, gradient):
        relu_derivative = (self.input > 0).astype(float)  # Derivácia ReLU
        return gradient * relu_derivative


class MSE(Module):
    def __init__(self):
        self.y_true = None  # Uchováva pravdivé hodnoty
        self.y_pred = None  # Uchováva predikované hodnoty

    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n = self.y_true.shape[0]
        # Výpočet MSE (Mean Squared Average)
        return (2 / n) * (self.y_pred - self.y_true)
    
# Funkcia na tréning
def train_network(problem_name, X, y, model, learning_rate, epochs, momentum):
    mse = MSE()
    # Uchováva priebeh straty
    losses = []

    for epoch in range(epochs):
        sum_loss = 0.0
        for i in range(len(X)):
            x_i = X[i:i+1]
            y_i = y[i:i+1]

            prediction = model.forward(x_i)
            loss = mse.forward(prediction, y_i)
            sum_loss += loss

            grad_loss = mse.backward()
            model.backward(grad_loss)
            model.updateweights(learning_rate, momentum)
        
        average_loss = sum_loss / len(X)
        losses.append(average_loss)

        # Výpis každých 100 epoch pre sledovanie priebehu
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"{"-"*40}")
            print(f"Epoch c. {epoch + 1}, Priemerna strata: {average_loss:.4f}")
            predictions = model.forward(X)
            predictions_binary = (predictions > 0.5).astype(int)
            print(f"Predpoved {problem_name}:", predictions_binary.flatten())
            print(f"Hodnoty {problem_name}:", y.flatten())
    return losses

# Vizualizácia tréningovej chyby
def visual(rows,columns, pltidx, activation, lr, momentum, losses):
    plt.subplot(rows, columns, pltidx)
    plt.plot(losses, label="Trenovaci Loss", color='Purple')
    plt.title(f"AF: {activation}, LR: {lr}, M: {momentum}", fontsize=10)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.ticklabel_format(style='plain', axis='y')  # Zakáže vedecký zápis na osi y

# Funkcia na dynamickú stavbu modelu
def add_layers(hidden_layers=1, hidden_size=4, activation="relu"):
    layers = []
    inputsize = 2
    for lyr in range(hidden_layers):
        layers.append(Linear(inputsize, hidden_size, activation))
        if activation == "relu":
            layers.append(ReLU())
        elif activation == "sigmoid":
            layers.append(Sigmoid())
        elif activation == "tanh":
            layers.append(Tanh())
        else:
            print(f"Chybna aktivacna funkcia: {activation}")
        inputsize = hidden_size
    layers.append(Linear(inputsize, 1, activation="sigmoid"))
    layers.append(Sigmoid())

    return Model(layers)

# Konfigurácia testovacieho procesu
test_config = {
    "epochs" : 500,
    "lr_list" : [0.1, 0.05, 0.01],
    "momentum_list" : [0, 0.9],
    "activ_list" : ["sigmoid", "tanh", "relu"],
    "problem_list" : ["AND","OR","XOR"],
}

# AND a OR dáta
problem_datasets = {
    # XOR: Nelineárne separovateľný problém
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])),
    # AND: Lineárne separovateľný problém
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [0], [0], [1]])),
    # OR: Tiež lineárne separovateľný problém
    "OR":  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [1]])),
}

# Hlavná slučka
while True:
    problem_select = int(input("Vyberte problem:\n1 - XOR problem\n2- AND problem\n3 - OR problem\n>:"))
    print(f"\n{"-"*100}")

    # Nastavenie problémových dát
    if problem_select==1:
        print(">>> XOR problem <<<")
        problem_name = "XOR"
        X, y = problem_datasets[problem_name]
    elif problem_select==2:
        print(">>> AND problem <<<")
        problem_name = "AND"
        X, y = problem_datasets[problem_name]
    elif problem_select==3:
        print(">>> OR problem <<<")
        problem_name = "OR"
        X, y = problem_datasets[problem_name]
    else:
        print("Chybny vstup, ukoncenie programu!")
        exit()

    #test_case=int(input("Vlastny konfig/Testovaci konfig? (0/1)")) or 1

    rows = len(test_config["activ_list"])
    columns = len(test_config["lr_list"]) * len(test_config["momentum_list"])
    epochs= test_config["epochs"]
    plt_id = 1

    win_figure = plt.figure(figsize=(20, 15))
    win_figure.canvas.manager.set_window_title(f"{problem_name} problem")

    for activation in test_config["activ_list"]:
        for lr in test_config["lr_list"]:
            for momentum in test_config["momentum_list"]:
                print(f"{"-"*100}")
                print(f"Zvoleny {problem_name} problem => aktivacna funkcia = {activation}, rychlost ucenia = {lr}, momentum = {momentum} ~")
                
                model = add_layers(hidden_layers=1, hidden_size=4, activation=activation)
                losses=train_network(problem_name, X, y, model, lr, epochs, momentum)
                visual(rows, columns, plt_id, activation, lr, momentum, losses)
                plt_id+=1

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.suptitle(f"Trenovacia strata pre {problem_name} problem", fontsize=15, fontweight='bold')
    plt.show()
