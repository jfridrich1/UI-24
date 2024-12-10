import numpy as np
import matplotlib.pyplot as plt

class Module:
    def forward(self, input):
        raise NotImplementedError("Forward musí byť implementovaný.")

    def backward(self, gradient):
        raise NotImplementedError("Backward musí byť implementovaný.")
    def updateweights(this, learning_rate, momentum=0):#update parameters
        pass

class Model:
    def __init__(self, modules):
        #self.modules = []  # Zoznam modulov
        self.modules = modules
    
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
    def updateweights(this, learning_rate, momentum=0):
        for module in this.modules:
            if isinstance(module, Linear):
                module.updateweights(learning_rate, momentum)


class Linear(Module):
    """def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Malé náhodné váhy
        self.bias = np.zeros((1, output_size))  # Bias inicializovaný na nulu
        self.input = None  # Uchová vstupy z dopredného smeru
        self.gradient_weights = None  # Gradient váh
        self.gradient_bias = None  # Gradient biasu

        # Pridáme moment
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)"""
    def __init__(this, inputsize, outputsize, activation='relu'):
        #Inicializácia počiatočných vah a biasov, odporučil mi to čet4
        if activation == 'relu':
            this.W = np.random.randn(inputsize, outputsize) * np.sqrt(2 / inputsize)
        elif activation == 'tanh':
            this.W = np.random.randn(inputsize, outputsize) * np.sqrt(1 / inputsize)
        else:
            this.W = np.random.randn(inputsize, outputsize) * 0.01

        this.W = np.random.randn(inputsize, outputsize) * np.sqrt(1 / inputsize)
        this.b = np.zeros((1, outputsize))

        this.grad_W = np.zeros_like(this.W)
        this.grad_b = np.zeros_like(this.b)

        this.velocity_W = np.zeros_like(this.W)
        this.velocity_b = np.zeros_like(this.b)

    def forward(self, input):
        self.input = input  # Ukladáme vstup pre spätné šírenie
        return np.dot(input, self.weights) + self.bias

    def backward(self, gradient):
        # Gradienty váh a biasu
        self.gradient_weights = np.dot(self.input.T, gradient)
        self.gradient_bias = np.sum(gradient, axis=0, keepdims=True)

        # Gradient vzhľadom na vstup
        return np.dot(gradient, self.weights.T)
    

    def update_weights(self, learning_rate, beta=0.9):
        # Aktualizácia momenta pre váhy
        self.velocity_weights = beta * self.velocity_weights - learning_rate * self.gradient_weights
        self.weights += self.velocity_weights

        # Aktualizácia momenta pre biasy
        self.velocity_bias = beta * self.velocity_bias - learning_rate * self.gradient_bias
        self.bias += self.velocity_bias

    
class Sigmoid(Module):
    def __init__(self):
        self.output = None  # Ukladáme výstup z dopredného smeru pre deriváciu

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))  # Výpočet sigmoid
        return self.output

    def backward(self, gradient):
        sigmoid_derivative = self.output * (1 - self.output)  # Derivácia sigmoid
        return gradient * sigmoid_derivative


class Tanh(Module):
    def __init__(self):
        self.output = None  # Uchová výstup z dopredného smeru

    def forward(self, input):
        self.output = np.tanh(input)  # Výpočet tanh
        return self.output

    def backward(self, gradient):
        tanh_derivative = 1 - self.output ** 2  # Derivácia tanh
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
        return (2 / n) * (self.y_pred - self.y_true)
    
# Funkcia na tréning
def train_network(problem_name, X, y, model, learning_rate, epochs, momentum):
    # Pridanie vrstiev
    """for layer in layers:
        model.add_module(Linear(layer[0], layer[1]))
        model.add_module(Sigmoid())"""

    # Chybová funkcia
    mse = MSE()
    losses = []

    for epoch in range(epochs):
        # Dopredný smer
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

        # Aktualizácia váh
        """for module in model.modules:
            if isinstance(module, Linear):
                if use_momentum:
                    module.update_weights(learning_rate, beta)
                else:
                    module.weights -= learning_rate * module.gradient_weights
                    module.bias -= learning_rate * module.gradient_bias"""


    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs} /Loss: {average_loss:.5f}')
        predictions = model.forward(X)
        predictions_binary = (predictions > 0.5).astype(int)
        print(f'Predictions of {problem_name}:', predictions_binary.flatten())
        print(f'Real values of {problem_name}:', y.flatten())
    return model, losses

# Vizualizácia tréningovej chyby
def visual(rows,columns, plotidx, activation, lr, momentum, losses):
    plt.subplot(rows, columns, plotidx)
    plt.plot(losses, label='Training Loss')
    plt.title(f'{activation}, LR: {lr}, Mom: {momentum}', fontsize=7)
    plt.xlabel('Epoch', fontsize=7)
    plt.ylabel('MSE Loss', fontsize=7)
    plt.grid(True)
    pltidx += 1
    plt.tick_params(axis='both', which='major', labelsize=7)

def graph(hidden_layers=1, hidden_size=4, activation='tanh'):
    layers = []
    inputsize = 2
    for _ in range(hidden_layers):
        layers.append(Linear(inputsize, hidden_size, activation))
        if activation == 'sigmoid':
            layers.append(Sigmoid())
        elif activation == 'tanh':
            layers.append(Tanh())
        elif activation == 'relu':
            layers.append(ReLU())
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        inputsize = hidden_size
    layers.append(Linear(inputsize, 1, activation='sigmoid'))
    layers.append(Sigmoid())
    return Model(layers)

# AND a OR dáta
datasets = {
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])),
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [0], [0], [1]])),
    "OR":  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [1]])),
}

epochs = 500
lr_list = [0.1, 0.05, 0.01]
momentum_list = [0, 0.9]
activ_list = ['sigmoid', 'tanh', 'relu']
problem_list = ['AND','OR','XOR']

# Definícia modelu
"""model = Model()
model.add_module(Linear(2, 4))  # Skrytá vrstva (2 vstupy, 4 neuróny)
model.add_module(Sigmoid())     # Aktivačná funkcia pre skrytú vrstvu
model.add_module(Linear(4, 1))  # Výstupná vrstva (4 vstupy, 1 výstup)
model.add_module(Sigmoid())     # Aktivačná funkcia pre výstup
"""
# Chybová funkcia
mse = MSE()


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

    #test_case=int(input("Vlastny konfig/Testovaci konfig? (0/1)")) or 1

    plt.figure(figsize=(20, 16))
    numofactivations = len(activ_list)
    numoflearningrates = len(lr_list)
    numofmomentums = len(momentum_list)
    rows = numofactivations
    columns = numoflearningrates * numofmomentums
    plotidx = 1

    for activation in activ_list:
        for lr in lr_list:
            for momentum in momentum_list:
                print(f'\nTrenovanie {problem_name} problem: aktivacna funkcia {activation}, rychlost ucenia: {lr}, momentum: {momentum}')
                model = graph(hidden_layers=1, hidden_size=4, activation=activation)
                mse = MSE()
                losses=train_network(problem_name, X, y, model, lr, epochs, momentum)
                visual(rows, columns, plotidx, activation, lr, momentum, losses)

    plt.suptitle(f'Training Loss of {problem_name} Problem', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
