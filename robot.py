import math, numpy as np, pygame #noqa F401
from neural_net import NeuroNet
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
        # Рисуем линию, показывающую направление робота
        line_length = 30
        line_end_x = self.x + line_length * np.cos(self.angle)
        line_end_y = self.y + line_length * np.sin(self.angle)
        pygame.draw.line(screen, (0, 0, 255), (int(self.x), int(self.y)), (int(line_end_x), int(line_end_y)), 2)

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
                return 1  # Увеличиваем счетчик собранного мусора
        return 0

class LearningRobot(Robot):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.brain = NeuroNet()  # Нейронная сеть для принятия решений
        self.last_distance = None  # Для хранения предыдущего расстояния до мусора

    def update(self, trashes):
        # Найти ближайший мусор
        distance, angle = self.find_closest_trash(trashes)

        # Нормализуем входные данные для нейронной сети
        inputs = np.array([distance, angle]).reshape(1, -1) / np.array([800, 180])

        # Используем нейронную сеть для предсказания скоростей моторов
        motor_speeds = self.brain.predict(inputs)

        # Преобразуем выходы сети в скорости для моторов
        left_motor_speed = motor_speeds[0, 0]
        right_motor_speed = motor_speeds[0, 1]

        # Обновляем положение робота с учетом скоростей
        super().update(left_motor_speed, right_motor_speed)

        # Если это не первый шаг, оцениваем награду за приближение к мусору
        if self.last_distance is not None:
            reward = self.calculate_reward(distance)
            self.train_network(reward)

        # Обновляем предыдущее расстояние
        self.last_distance = distance

    def calculate_reward(self, current_distance):  # sourcery skip: assign-if-exp
        """Возвращает награду в зависимости от того, приближается ли робот к мусору"""
        if current_distance < self.last_distance:
            return 0.1  # Небольшая положительная награда за приближение к мусору
        else:
            return -0.1  # Отрицательная награда за отдаление от мусора

    def collect_trash(self, trashes):
        # Если робот собрал мусор, возвращаем вознаграждение и обучаем сеть
        collected = super().collect_trash(trashes)
        if collected:
            # Обучаем сеть с наградой за сбор мусора
            self.train_network(1)  # Награда за сбор мусора

        # Если мусор не собран, продолжаем обучение с наградой за направление
        else:
            self.train_network(0)  # Награды нет

        return collected

    def train_network(self, reward):
        """Обучает сеть на основе награды"""
        # Если есть данные для тренировки
        if self.last_distance is not None:
            inputs = np.array([self.last_distance, reward]).reshape(1, -1)
            expected_output = np.array([[reward, reward]])  # Ожидаемая награда
            self.brain.train(inputs, expected_output)  # Тренируем сеть