import pygame
import numpy as np

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0.1
        self.size = 20

    def update(self, left_motor_speed, right_motor_speed):
        # Определяем изменения в движении на основе скорости моторов
        self.angle += (right_motor_speed - left_motor_speed) * 0.05
        self.x += self.speed * (left_motor_speed + right_motor_speed) * np.cos(self.angle)
        self.y += self.speed * (left_motor_speed + right_motor_speed) * np.sin(self.angle)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

# Инициализация Pygame и робота
pygame.init()
screen = pygame.display.set_mode((800, 600))
robot = Robot(400, 300)

running = True
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Обновление и отрисовка робота
    robot.update(0.5, 0.5)
    robot.draw(screen)
    pygame.display.flip()
