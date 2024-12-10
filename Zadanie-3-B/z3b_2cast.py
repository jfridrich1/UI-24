import numpy as np
import matplotlib.pyplot as plt

class Module:
    def forward(self, input):
        raise NotImplementedError("Forward musí byť implementovaný.")

    def backward(self, gradient):
        raise NotImplementedError("Backward musí byť implementovaný.")

class Model:
    def __init__(self):
        self.modules = []  # Zoznam modulov
    
    def add_module(self, module):
        self.modules.append(module)
    
    def forward(self, input):
        for module in self.modules:
            input = module.forward(input)
        return input
    
    def backward(self, gradient):
        """
        Vykoná spätné šírenie cez všetky moduly (v opačnom poradí).
        """
        for module in reversed(self.modules):
            gradient = module.backward(gradient)


class Linear(Module):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Malé náhodné váhy
        self.bias = np.zeros((1, output_size))  # Bias inicializovaný na nulu
        self.input = None  # Uchová vstupy z dopredného smeru
        self.gradient_weights = None  # Gradient váh
        self.gradient_bias = None  # Gradient biasu

        # Pridáme moment
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    def forward(self, input):
        """
        Dopredné šírenie: y = xW + b
        """
        self.input = input  # Ukladáme vstup pre spätné šírenie
        return np.dot(input, self.weights) + self.bias

    def backward(self, gradient):
        # Gradienty váh a biasu
        self.gradient_weights = np.dot(self.input.T, gradient)
        self.gradient_bias = np.sum(gradient, axis=0, keepdims=True)

        # Gradient vzhľadom na vstup
        return np.dot(gradient, self.weights.T)
    

    def update_weights(self, learning_rate, beta=0.9):
        """
        Aktualizuje váhy s momentom.
        - learning_rate: rýchlosť učenia.
        - beta: koeficient momenta.
        """
        # Aktualizácia momenta pre váhy
        self.velocity_weights = beta * self.velocity_weights - learning_rate * self.gradient_weights
        self.weights += self.velocity_weights

        # Aktualizácia momenta pre biasy
        self.velocity_bias = beta * self.velocity_bias - learning_rate * self.gradient_bias
        self.bias += self.velocity_bias

    
class Sigmoid(Module):
    def __init__(self):
        """
        Sigmoid funkcia je bezparametrická, nie sú potrebné váhy ani bias.
        """
        self.output = None  # Ukladáme výstup z dopredného smeru pre deriváciu

    def forward(self, input):
        """
        Dopredné šírenie: sigmoid(x).
        """
        self.output = 1 / (1 + np.exp(-input))  # Výpočet sigmoid
        return self.output

    def backward(self, gradient):
        """
        Spätné šírenie: výpočet gradientu.
        - gradient: gradient chyby vzhľadom na výstup tejto vrstvy.
        """
        sigmoid_derivative = self.output * (1 - self.output)  # Derivácia sigmoid
        return gradient * sigmoid_derivative


class Tanh(Module):
    def __init__(self):
        """
        Tanh funkcia je bezparametrická, nie sú potrebné váhy ani bias.
        """
        self.output = None  # Uchová výstup z dopredného smeru

    def forward(self, input):
        """
        Dopredné šírenie: tanh(x).
        """
        self.output = np.tanh(input)  # Výpočet tanh
        return self.output

    def backward(self, gradient):
        """
        Spätné šírenie: výpočet gradientu.
        - gradient: gradient chyby vzhľadom na výstup tejto vrstvy.
        """
        tanh_derivative = 1 - self.output ** 2  # Derivácia tanh
        return gradient * tanh_derivative


class ReLU(Module):
    def __init__(self):
        """
        ReLU funkcia je bezparametrická, nie sú potrebné váhy ani bias.
        """
        self.input = None  # Uchová vstup z dopredného smeru

    def forward(self, input):
        """
        Dopredné šírenie: ReLU(x).
        """
        self.input = input
        return np.maximum(0, input)  # Výpočet ReLU

    def backward(self, gradient):
        """
        Spätné šírenie: výpočet gradientu.
        - gradient: gradient chyby vzhľadom na výstup tejto vrstvy.
        """
        relu_derivative = (self.input > 0).astype(float)  # Derivácia ReLU
        return gradient * relu_derivative


class MSE(Module):
    def __init__(self):
        """
        MSE funkcia nemá žiadne váhy alebo bias.
        """
        self.y_true = None  # Uchováva pravdivé hodnoty
        self.y_pred = None  # Uchováva predikované hodnoty

    def forward(self, y_pred, y_true):
        """
        Dopredné šírenie: výpočet MSE.
        - y_pred: predikované hodnoty.
        - y_true: skutočné hodnoty.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        """
        Spätné šírenie: výpočet gradientu MSE.
        """
        n = self.y_true.shape[0]
        return (2 / n) * (self.y_pred - self.y_true)
    
# Funkcia na tréning
def train_network(data, layers, learning_rate, epochs, use_momentum=False, beta=0.9):
    X, y = data
    model = Model()

    # Pridanie vrstiev
    for layer in layers:
        model.add_module(Linear(layer[0], layer[1]))
        model.add_module(Sigmoid())

    # Chybová funkcia
    mse = MSE()
    losses = []

    for epoch in range(epochs):
        # Dopredný smer
        output = model.forward(X)
        loss = mse.forward(output, y)
        losses.append(loss)

        # Spätný smer
        loss_gradient = mse.backward()
        model.backward(loss_gradient)

        # Aktualizácia váh
        for module in model.modules:
            if isinstance(module, Linear):
                if use_momentum:
                    module.update_weights(learning_rate, beta)
                else:
                    module.weights -= learning_rate * module.gradient_weights
                    module.bias -= learning_rate * module.gradient_bias

    return model, losses

# Vizualizácia tréningovej chyby
def visual(losses, title):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



# XOR dáta
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Vstupy
#y = np.array([[0], [1], [1], [0]])  # Očakávané výstupy

# AND a OR dáta
datasets = {
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])),
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [0], [0], [1]])),
    "OR":  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [1]])),
}

# Definícia modelu
model = Model()
model.add_module(Linear(2, 4))  # Skrytá vrstva (2 vstupy, 4 neuróny)
model.add_module(Sigmoid())     # Aktivačná funkcia pre skrytú vrstvu
model.add_module(Linear(4, 1))  # Výstupná vrstva (4 vstupy, 1 výstup)
model.add_module(Sigmoid())     # Aktivačná funkcia pre výstup

# Chybová funkcia
mse = MSE()

# Tréning parametre
learning_rate = 0.1
epochs = 500


while True:
    problem_select = int(input("Vyberte problem:\n1 - AND problem\n2- OR problem\n3 - XOR problem\n>:"))

    if problem_select==1:
        print("\nAND problem:")
        problem_name = 'AND'
        X, y = datasets[problem_name]
    elif problem_select==2:
        print("\nOR problem:")
        problem_name = 'OR'
        X, y = datasets[problem_name]
    elif problem_select==3:
        print("\nXOR problem:")
        problem_name = 'XOR'
        X, y = datasets[problem_name]
    else:
        print("Invalid choice")
        exit()
        continue

# Výstupy po tréningu
print("\nVýsledky po tréningu:")
for i in range(len(X)):
    print(f"Vstup: {X[i]}, Očakávané: {y[i]}, Predpovedané: {output[i][0]:.3f}")
