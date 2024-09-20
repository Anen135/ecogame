import math, numpy as np, pygame #noqa F401
from neural_net import NeuralNetwork
from tqdm import tqdm

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 1
        self.size = 20
        self.collected_trash = 0  # Счетчик собранного мусора

    def update(self, left_motor_speed, right_motor_speed):
        self.angle += (right_motor_speed - left_motor_speed) * 0.05
        self.x += self.speed * (left_motor_speed + right_motor_speed) * np.cos(self.angle)
        self.y += self.speed * (left_motor_speed + right_motor_speed) * np.sin(self.angle)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

    def find_closest_trash(self, trashes):
        closest_trash = None
        min_distance = float('inf')
        direction = 0

        for trash in trashes:
            # Вычисляем расстояние до каждого объекта мусора
            distance = math.sqrt((trash.x - self.x) ** 2 + (trash.y - self.y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_trash = trash

        if closest_trash:
            # Вычисляем угол между роботом и мусором
            dx = closest_trash.x - self.x
            dy = closest_trash.y - self.y
            direction = math.atan2(dy, dx) - self.angle
            direction = math.degrees(direction)

        return min_distance, direction

    def collect_trash(self, trashes):
        # Проверяем, находится ли робот в радиусе любого мусора
        for trash in trashes[:]:
            distance = math.sqrt((trash.x - self.x) ** 2 + (trash.y - self.y) ** 2)
            if distance < self.size + trash.size:  # Если робот соприкоснулся с мусором
                trashes.remove(trash)
                self.collected_trash += 1  # Увеличиваем счетчик собранного мусора

class RobotWithNN(Robot):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.nn = self.train()

    def update(self, trashes):
        # Получаем информацию о ближайшем мусоре
        distance, direction = self.find_closest_trash(trashes)
        
        # Подготавливаем входные данные для нейронной сети
        inputs = [self.x, self.y, distance, direction]
        
        # Получаем выходные данные от сети (управление моторами)
        motor_speeds = self.nn.forward(inputs)
        
        # Используем эти данные для обновления скорости моторов
        left_motor_speed, right_motor_speed = motor_speeds
        
        # Ограничиваем значения скоростей для моторов
        left_motor_speed = max(min(left_motor_speed, 1), -1)
        right_motor_speed = max(min(right_motor_speed, 1), -1)
        
        # Обновляем положение робота
        super().update(left_motor_speed, right_motor_speed)
    
    def train(self):
        # Пример обучающих данных (упрощенные для демонстрации)
        training_data = [
            {"inputs": [400, 300, 50, 30], "outputs": [0.5, 0.5]},   # Робот далеко от мусора, оба мотора движутся вперед
            {"inputs": [100, 200, 20, -45], "outputs": [1, 0.2]},    # Мусор слева, нужно поворачивать направо
            {"inputs": [600, 500, 10, 45], "outputs": [0.2, 1]},     # Мусор справа, нужно поворачивать налево
            {"inputs": [200, 100, 100, 0], "outputs": [0.7, 0.7]},   # Мусор впереди, оба мотора движутся вперед
        ]

        # Обучаем нейронную сеть с 4 входами, 5 скрытыми нейронами и 2 выходами
        robot_nn = NeuralNetwork(num_inputs=4, num_hidden=5, num_outputs=2)

        for _ in tqdm(range(1000)):
            for data in training_data:
                inputs = np.array(data['inputs'])
                expected_outputs = np.array(data['outputs'])
                robot_nn.train(inputs, expected_outputs, learning_rate=0.01)
        return robot_nn