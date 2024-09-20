import math, numpy as np, pygame #noqa F401

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

