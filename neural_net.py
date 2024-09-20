import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation='relu'):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activate(z)

    def activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        # Можно добавить другие активационные функции

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # Создаем скрытый слой из num_hidden нейронов
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden)]
        
        # Создаем выходной слой из num_outputs нейронов
        self.output_layer = [Neuron(num_hidden) for _ in range(num_outputs)]
        
    def forward(self, inputs):
        # Прогоняем входные данные через скрытый слой
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        
        # Прогоняем результаты через выходной слой
        outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return outputs
