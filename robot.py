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
        self.last_angle = None  # Для хранения предыдущего угла до мусора
        self.last_distance = None  # Для хранения предыдущего расстояния до мусора
        self.training_step = 0     # Счётчик шагов для управления частотой обучения
        self.training_interval = 10  # Проводим обучение каждые 10 шагов
        self.penalty_count = 0  # Счётчик штрафов для усиления поворота

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

        # Усиление поворота, если робот получил несколько штрафов подряд
        if self.penalty_count > 10:
            # Увеличиваем разницу в скоростях моторов для более агрессивного поворота
            left_motor_speed -= 0.5  # Замедляем левый мотор
            right_motor_speed += 0.5  # Ускоряем правый мотор
            print("Усиленный поворот робота из-за штрафа")

        # Обновляем положение робота с учетом скоростей
        super().update(left_motor_speed, right_motor_speed)

        # Оцениваем награду за движение в сторону мусора (если угол уменьшается)
        if self.last_angle is not None and self.last_distance is not None:
            reward = self.calculate_angle_distance_reward(angle, distance)

            # Проводим обучение каждые `training_interval` шагов
            self.training_step += 1
            if self.training_step % self.training_interval == 0:
                self.train_network(reward)

        # Обновляем предыдущее значение угла и расстояния
        self.last_angle = angle
        self.last_distance = distance

    def calculate_angle_distance_reward(self, current_angle, current_distance):
        """
        Возвращает награду за движение к мусору:
        - Максимальная награда (1), если угол близок к 0 и расстояние уменьшается.
        - Нейтральная награда (0), если угол 90 градусов (робот движется перпендикулярно или не двигается к мусору).
        - Штраф (-1), если угол 180 градусов или расстояние увеличивается.
        """
        # Если робот смотрит прямо на мусор и расстояние до него уменьшается
        if abs(current_angle) % 360 <= 30 and current_distance < self.last_distance:
            self.penalty_count = 0  # Сброс штрафов, если робот движется в правильную сторону
            return 1  # Максимальная награда за движение к мусору
        # Если робот смотрит на мусор, но не движется к нему (расстояние не уменьшается)
        elif abs(current_angle) % 360 <= 120 and current_distance >= self.last_distance:
            return 0.5  # Умеренная награда за направление на мусор, но без движения
        # Если робот движется в противоположную сторону
        else:
            self.penalty_count += 1  # Увеличиваем счётчик штрафов
            return -1  # Штраф за неправильное направление и/или отдаление от мусора

    def collect_trash(self, trashes):
        # Если робот собрал мусор, возвращаем вознаграждение и обучаем сеть
        collected = super().collect_trash(trashes)
        if collected:
            # Обучаем сеть с наградой за сбор мусора
            self.train_network(1)  # Награда за сбор мусора
            self.penalty_count = 0  # Сброс штрафов при успешном сборе мусора

        return collected

    def train_network(self, reward):
        """Обучает сеть на основе награды"""
        # Если есть данные для тренировки
        if self.last_angle is not None:
            inputs = np.array([0, self.last_angle]).reshape(1, -1)  # Мы тренируемся по углу
            expected_output = np.array([[reward, reward]])  # Ожидаемая награда
            self.brain.train(inputs, expected_output, iterations=100)  # Тренируем сеть
