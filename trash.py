import pygame

class Trash:
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.size = size

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.size)
