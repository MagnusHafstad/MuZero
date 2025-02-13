import pygame
import pygame_gui
import numpy as np


class SnakeGUI:
    def __init__(self, matrix_size):
        pygame.init()
        self.matrix_size = matrix_size
        self.cell_size = 60
        self.play_grid = (matrix_size*self.cell_size, matrix_size*self.cell_size)
        self.color_map = {0:(255,255,255), 1: (0,255,0),2: (255,0,0)} 
        self.window_surface = pygame.display.set_mode(self.play_grid)

        self.manager = pygame_gui.UIManager(self.play_grid)

        self.background = pygame.Surface(self.play_grid)
        self.background.fill(pygame.Color(self.color_map[0]))

    def update_gui(self, matrix):
        for row in range(self.matrix_size):
                for col in range(self.matrix_size):
                    color = self.color_map[matrix[row][col]]
                    pygame.draw.rect(self.window_surface, color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def user_input(self, prev_direction):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_UP:
                        return "up"
                    case pygame.K_DOWN:
                        return "down"
                    case pygame.K_LEFT:
                        return "left"
                    case pygame.K_RIGHT:
                        return "right"
        return prev_direction



if __name__ == "__main__":
     snake = SnakeGUI(8)
     snake.run()