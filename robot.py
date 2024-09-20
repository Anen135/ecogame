import math, numpy as np, pygame #noqa F401
from neural_net import NeuralNetwork
from tqdm import tqdm

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0.1
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
            {"inputs": [300, 200, 15, 0], "outputs": [0.8, 0.8]},  # Мусор близко, оба мотора движутся вперед
            {"inputs": [50, 100, 5, -60], "outputs": [1.0, 0.1]},  # Мусор очень близко слева, сильный поворот направо
            {"inputs": [500, 400, 8, 60], "outputs": [0.1, 1.0]},  # Мусор очень близко справа, сильный поворот налево
            {"inputs": [150, 150, 30, -15], "outputs": [0.9, 0.4]},  # Мусор слева, небольшой поворот направо
            {"inputs": [450, 450, 25, 15], "outputs": [0.4, 0.9]},  # Мусор справа, небольшой поворот налево
            {"inputs": [100, 100, 10, -90], "outputs": [1.0, 0.0]},  # Мусор очень близко слева, резкий поворот направо
            {"inputs": [500, 500, 10, 90], "outputs": [0.0, 1.0]},  # Мусор очень близко справа, резкий поворот налево
            {"inputs": [200, 200, 50, 0], "outputs": [0.6, 0.6]},   # Мусор далеко, но по центру, оба мотора движутся вперед
            {"inputs": [200, 200, 10, 0], "outputs": [0.9, 0.9]},   # Мусор близко, по центру, оба мотора движутся вперед
            {"inputs": [200, 200, 100, 0], "outputs": [0.8, 0.8]},  # Мусор близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [300, 300, 10, 0], "outputs": [0.9, 0.9]},  # Мусор близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [100, 100, 10, -30], "outputs": [1.0, 0.4]},  # Мусор слева, небольшой поворот направо
            {"inputs": [400, 400, 10, 30], "outputs": [0.4, 1.0]},  # Мусор справа, небольшой поворот налево
            {"inputs": [250, 250, 10, -10], "outputs": [0.9, 0.7]},  # Мусор слева, очень слабый поворот направо
            {"inputs": [350, 350, 10, 10], "outputs": [0.7, 0.9]},  # Мусор справа, очень слабый поворот налево
            {"inputs": [200, 200, 20, 0], "outputs": [0.8, 0.8]},  # Мусор близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [200, 200, 30, 0], "outputs": [0.9, 0.9]},  # Мусор близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [200, 200, 40, 0], "outputs": [0.95, 0.95]},  # Мусор близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [200, 200, 5, 0], "outputs": [0.7, 0.7]},   # Мусор очень близко, по центру, оба мотора движутся вперед
            {"inputs": [100, 100, 5, 0], "outputs": [0.6, 0.6]},   # Мусор очень близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [50, 50, 5, 0], "outputs": [0.5, 0.5]},   # Мусор очень близко, по центру, оба мотора движутся вперед (вариация)
            {"inputs": [200, 200, 10, -5], "outputs": [0.9, 0.8]},  # Мусор близко, чуть левее центра, оба мотора движутся вперед
            {"inputs": [200, 200, 10, 5], "outputs": [0.8, 0.9]},  # Мусор близко, чуть правее центра, оба мотора движутся вперед
        ]

        # Обучаем нейронную сеть с 4 входами, 10 скрытыми нейронами и 2 выходами
        robot_nn = NeuralNetwork(num_inputs=4, num_hidden=10, num_outputs=2)

        for _ in tqdm(range(10000)):
            for data in training_data:
                inputs = np.array(data['inputs'])
                expected_outputs = np.array(data['outputs'])
                robot_nn.train(inputs, expected_outputs, learning_rate=0.0001)
        return robot_nn

class QLearningRobot(Robot):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.q_table = np.zeros((800, 600, 360, 4))  # Пример Q-таблицы с 4 возможными действиями
        self.alpha = 0.9  # Скорость обучения
        self.gamma = 0.9  # Коэффициент дисконтирования будущих наград
        self.epsilon = 0.1  # Вероятность случайного действия для эпсилон-жадной стратегии
        self.actions = [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.0, 0.0)]  # Действия: вперед, влево, вправо, остановка
        self.reward = None

    def get_state(self):
        # Преобразуем текущее положение и угол в состояние
        x_state = min(int(self.x), 799)  # Ограничиваем x от 0 до 799
        y_state = min(int(self.y), 599)  # Ограничиваем y от 0 до 599
        angle_state = int(self.angle % 359)  # Угол в диапазоне от 0 до 359
        return x_state, y_state, angle_state

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Случайное действие (исследование)
            return np.random.randint(0, 4)
        else:
            # Выбираем действие с максимальным значением Q
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        # Обновление Q-значения с использованием формулы Q-learning
        best_future_q = np.max(self.q_table[new_state])  # Используем новое состояние
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * best_future_q - current_q)

    def get_reward(self, old_distance, new_distance, collected_trash):
        if collected_trash:
            return +10  # Большая награда за сбор мусора
        elif new_distance < old_distance:
            return +10  # Награда за приближение к мусору
        else:
            return -5  # Штраф за удаление от мусора

    def update(self, trashes):
        # Получаем текущее состояние
        state = self.get_state()

        # Находим ближайший мусор и вычисляем расстояние и направление до него
        distance, direction = self.find_closest_trash(trashes)

        # Выбираем действие на основе текущего состояния
        action = self.choose_action(state)

        # Применяем действие (управляем моторами)
        left_motor_speed, right_motor_speed = self.actions[action]
        super().update(left_motor_speed, right_motor_speed)

        # Проверяем сбор мусора и получаем новую информацию о расстоянии
        collected_trash = self.collect_trash(trashes)
        new_distance, _ = self.find_closest_trash(trashes)

        # Получаем новое состояние
        new_state = self.get_state()

        # Рассчитываем награду на основе изменений
        self.reward = self.get_reward(distance, new_distance, collected_trash)

        # Обновляем Q-таблицу

        self.update_q_table(state, action, self.reward, new_state)
    def manual_reward(self, reward):
        # Если игрок выдает награду, обновляем Q-таблицу с помощью вручную заданного значения
        if self.last_state is not None and self.last_action is not None:
            new_state = self.get_state()
            self.update_q_table(self.last_state, self.last_action, reward, new_state)