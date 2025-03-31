import pygame
import pygame_gui
import numpy as np

import yaml

with open("config.yaml", "r") as fule:
    config = yaml.safe_load(fule)


class SnakeGUI:
    def __init__(self, matrix_size):
        pygame.init()
        self.matrix_size = matrix_size
        self.cell_size = 60
        self.play_grid = (matrix_size*self.cell_size, matrix_size*self.cell_size)
        self.color_map = {0: (255, 255, 255), 1: config["snake_color"], -1: (255, 0, 0)}
        self.window_surface = pygame.display.set_mode(self.play_grid)

        self.manager = pygame_gui.UIManager(self.play_grid)

        self.background = pygame.Surface(self.play_grid)
        self.background.fill(pygame.Color(self.color_map[0]))


    def update_gui(self, matrix):
        max_idx = np.argmax(matrix)
        row, col = np.unravel_index(max_idx, matrix.shape)
        max_value = matrix[row, col]
        self.color_map.update({max_value: config["snake_head_color"]})

        for row in range(self.matrix_size):
                for col in range(self.matrix_size):
                    color= self.color_map[matrix[row][col]]
                    pygame.draw.rect(self.window_surface, color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
        pygame.time.wait(100)
        self.color_map.update({max_value: config["snake_color"]})
        pygame.display.flip()

    def user_input(self, prev_direction):
        """
        0: right
        1: left
        2: up
        3: down
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_UP:
                        return 2
                    case pygame.K_DOWN:
                        return 3
                    case pygame.K_LEFT:
                        return  1
                    case pygame.K_RIGHT:
                        return 0
        return prev_direction



gui = SnakeGUI(5)