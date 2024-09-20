import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation='relu'):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs) * 0.01  # Меньшие начальные веса
        self.bias = np.random.randn() * 0.01
        self.activation = activation
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = np.asarray(inputs)
        z = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.activate(z)
        return self.output

    def activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))

    def activate_derivative(self, z):
        if self.activation == 'relu':
            return 1 if z > 0 else 0
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)

    def backward(self, dL_dout, learning_rate, l2_lambda=0.01, clip_value=1.0):
        z = np.dot(self.inputs, self.weights) + self.bias
        dz = dL_dout * self.activate_derivative(z)

        dw = np.dot(self.inputs.T, dz)
        db = dz

        # Клиппинг градиентов
        dw = np.clip(dw, -clip_value, clip_value)
        db = np.clip(db, -clip_value, clip_value)

        # Добавляем регуляризацию L2
        self.weights -= learning_rate * (dw + l2_lambda * self.weights)
        self.bias -= learning_rate * np.sum(db)

        # Ограничиваем значения весов и смещений
        self.weights = np.clip(self.weights, -1000, 1000)
        self.bias = np.clip(self.bias, -1000, 1000)

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # Скрытый слой с активацией ReLU
        self.hidden_layer = [Neuron(num_inputs, activation='relu') for _ in range(num_hidden)]
        
        # Выходной слой с активацией tanh
        self.output_layer = [Neuron(num_hidden, activation='tanh') for _ in range(num_outputs)]
        
    def forward(self, inputs):
        # Прогоняем входн```ые данные через скрытый слой
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        
        # Прогоняем результаты через выходной слой
        outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        
        return outputs
    
    def backward(self, inputs, expected_outputs, actual_outputs, learning_rate, l2_lambda=0.01):
        # Ошибка для выходного слоя
        output_errors = [expected_outputs[i] - actual_outputs[i] for i in range(len(expected_outputs))]

        # Backpropagation для выходного слоя
        for i, neuron in enumerate(self.output_layer):
            neuron.backward(output_errors[i], learning_rate, l2_lambda)

        # Ошибка для скрытого слоя
        hidden_errors = [0] * len(self.hidden_layer)
        for i, hidden_neuron in enumerate(self.hidden_layer):
            for j, output_neuron in enumerate(self.output_layer):
                hidden_errors[i] += output_errors[j] * output_neuron.weights[i]

        # Backpropagation для скрытого слоя
        for i, neuron in enumerate(self.hidden_layer):
            neuron.backward(hidden_errors[i], learning_rate, l2_lambda)

    def train(self, inputs, expected_outputs, learning_rate=0.01, l2_lambda=0.01):
        # Прямой проход
        actual_outputs = self.forward(inputs)

        # Обратный проход
        self.backward(inputs, expected_outputs, actual_outputs, learning_rate, l2_lambda)
