import pygame
from robot import Robot
from trash import Trash
from neural_net import NeuralNetwork

class RobotWithNN(Robot):
    def __init__(self, x, y):
        super().__init__(x, y)
        # Инициализируем нейронную сеть с 4 входами, 5 скрытыми нейронами и 2 выходами
        self.nn = NeuralNetwork(num_inputs=4, num_hidden=5, num_outputs=2)

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


def display_collected_trash(screen, font, count):
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
trashes = [Trash(500, 300), Trash(600, 300), Trash(700, 300)]

running = True
while running:
    screen.fill((0, 0, 0))

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if robot.collected_trash == 3:
        running = False
    # Обновление и отрисовка робота
    robot.update(trashes)
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