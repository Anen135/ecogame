import pygame
from robot import LearningRobot  # Импорт нового робота с нейронной сетью
from trash import Trash
from threading import Thread
import os, time  # noqa: E401

class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.font = pygame.font.SysFont("Arial", 30)
        self.collected_trash_count = 0
        self.clock = pygame.time.Clock()
        self.thread = Thread(target=self.print_robots_info)
        self.thread.daemon = True

        # Заменяем Robot на LearningRobot с нейронной сетью
        self.robot = LearningRobot(400, 300)
        self.trashes = [
            Trash(500, 300),
            Trash(600, 350),
            Trash(700, 400),
            Trash(100, 100),
            Trash(200, 200),
            Trash(150, 450),
            Trash(300, 500),
            Trash(550, 550),
            Trash(700, 150),
            Trash(250, 250),
        ]

    def display_collected_trash(self):
        # Рендерим текст для отображения количества собранного мусора
        self.screen.blit(self.font.render(f"Собрано мусора: {self.collected_trash_count}", True, (255, 255, 255)), (10, 10))
    
    def display_fps(self):
        # Рендерим FPS
        self.screen.blit(self.font.render(f"FPS: {self.clock.get_fps():.2f}", True, (255, 255, 255)), (10, 40))
        
    def print_robots_info(self):
        while True:
            os.system("cls")
            print(f"Робот: x={self.robot.x}, y={self.robot.y}, угол={self.robot.angle}\n Мусор: dist={self.robot.find_closest_trash(self.trashes)[0]}, dir={abs(self.robot.find_closest_trash(self.trashes)[1]) % 360}")
            time.sleep(0.5)
            
    
    def run(self):
        running = True
        while running:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Условие завершения игры, когда весь мусор собран
            if not self.trashes:
                running = False

            # Очистка экрана
            self.screen.fill((0, 0, 0))
            
            # Нахождение ближайшего мусора и обновление состояния робота
            self.robot.update(self.trashes)
            
            # Проверка сбора мусора и обновление счетчика
            self.collected_trash_count += self.robot.collect_trash(self.trashes)
            
            # Отрисовка робота
            self.robot.draw(self.screen)
            
            # Отрисовка оставшегося мусора
            for trash in self.trashes:
                trash.draw(self.screen)
            
            # Отображение количества собранного мусора
            self.display_collected_trash()
            self.display_fps()
            
            # Обновление экрана
            pygame.display.flip()
            
            # Ограничение FPS (например, 60 FPS)
            self.clock.tick(60)
        
        # Завершение работы Pygame
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.thread.start()
    game.run()
