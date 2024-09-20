import pygame
from robot import RobotWithNN
from trash import Trash
import math

def display_collected_trash(screen, font, count): # sourcery skip: instance-method-first-arg-name
    # Рендерим текст для отображения
    text = font.render(f"Собрано мусора: {count}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

# Инициализация Pygame и объектов
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Настройка шрифта для отображения GUI
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

# Создание объектов
robot = RobotWithNN(400, 300)
trashes = [Trash(500, 300), Trash(600, 350), Trash(700, 400)]

running = True
while running:
    if math.isnan(robot.x) or math.isnan(robot.y):
        robot = RobotWithNN(400, 300)
    screen.fill((0, 0, 0))

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if robot.collected_trash == 3:
        running = False
    # Обновление и отрисовка робота
    robot.update(trashes)
    print(f"[{robot.x} {robot.y}] {robot.angle} {robot.speed} {robot.collected_trash}")
    robot.draw(screen)

    # Проверка сбора мусора
    robot.collect_trash(trashes)

    # Отрисовка мусора
    for trash in trashes:
        trash.draw(screen)

    # Отображение количества собранного мусора
    display_collected_trash(screen, font, robot.collected_trash)

    # Обновляем экран
    pygame.display.flip()