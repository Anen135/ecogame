import pygame
from robot import QLearningRobot  # Импорт QLearningRobot
from trash import Trash
import os
import threading
import time

def display_collected_trash(screen, font, count):
    # Рендерим текст для отображения
    text = font.render(f"Собрано мусора: {count}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

def print_to_console(robot):
    while True:
        os.system('cls')
        print(f"[{round(robot.x, 2)} {round(robot.y, 2)}]    \tA:{round(robot.angle, 2)}\t   S:{round(robot.speed, 2)} Tc:{robot.collected_trash}\n robotA: {robot.choose_action(robot.get_state())}\t robotR: {robot.reward}")
        time.sleep(0.1)

# Инициализация Pygame и объектов
pygame.init()
screen = pygame.display.set_mode((800, 600))


# Настройка шрифта для отображения GUI
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

# Создание робота и мусора
robot = QLearningRobot(400, 300)
trashes = [Trash(500, 300), Trash(600, 350), Trash(700, 400), Trash(100, 100), Trash(200, 200),
           Trash(150, 450), Trash(300, 500), Trash(550, 550), Trash(700, 150), Trash(250, 250)]


# Запуск потока для отображения данных робота
thread = threading.Thread(target=print_to_console, args=(robot,))
thread.daemon = True
thread.start()

running = True
collected_trash_count = 0  # Счетчик собранного мусора
clock = pygame.time.Clock()  # Для контроля FPS

# Главный игровой цикл
while running:
    screen.fill((0, 0, 0))  # Очистка экрана

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    keys = pygame.key.get_pressed()
    if keys[pygame.K_n]:  # Нажатие клавиши "N" для положительной награды
        robot.manual_reward(100)  # Положительная награда
    elif keys[pygame.K_m]:  # Нажатие клавиши "M" для отрицательной награды
        robot.manual_reward(-100)  # Отрицательная награда

    # Если робот собрал весь мусор
    if robot.collected_trash == len(trashes):
        running = False

    # Обновление робота: обучение через взаимодействие с окружающей средой
    robot.update(trashes)
    
    # Отрисовка робота
    robot.draw(screen)

    # Проверка сбора мусора
    robot.collect_trash(trashes)

    # Если робот собрал мусор, увеличиваем счетчик
    collected_trash_count = robot.collected_trash

    # Отрисовка мусора
    for trash in trashes:
        trash.draw(screen)

    # Отображение количества собранного мусора на экране
    display_collected_trash(screen, font, collected_trash_count)

    # Обновляем экран
    pygame.display.flip()

    # Ограничение FPS (например, 60 FPS)
    clock.tick(60)

# Выход из Pygame
pygame.quit()
