import numpy as np

class NeuroNet:
    def __init__(self):
        # Инициализация сети с одним скрытым слоем
        self.syn0 = 2 * np.random.random((2, 4)) - 1  # Веса для входного слоя
        self.syn1 = 2 * np.random.random((4, 2)) - 1  # Веса для скрытого слоя
    
    def sigmoid(self, x, der=False):
        return x * (1 - x) if der else 1 / (1 + np.exp(-x))

    def train(self, inputs, expected_output, iterations=10000):
        for _ in range(iterations):
            # Прямое распространение
            l0 = inputs
            l1 = self.sigmoid(np.dot(l0, self.syn0))  # Скрытый слой
            l2 = self.sigmoid(np.dot(l1, self.syn1))  # Выходной слой

            # Ошибка выходного слоя
            l2_error = expected_output - l2
            l2_delta = l2_error * self.sigmoid(l2, True)

            # Ошибка скрытого слоя
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * self.sigmoid(l1, True)

            # Обновление весов
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)

        return l2

    def predict(self, inputs):
        # Прогнозирование на основе текущих весов
        l1 = self.sigmoid(np.dot(inputs, self.syn0))
        return self.sigmoid(np.dot(l1, self.syn1))

