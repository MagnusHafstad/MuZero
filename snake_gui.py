import pygame
import pygame_gui
import numpy as np

matrix_size = 10
cell_size = 60

play_grid = (matrix_size*cell_size, matrix_size*cell_size)

background_color =(255,0,0)
black = (0,0,0)

color_map = {0:(255,33,0), 1: (0,222,0),2: (0,0,222)} 


matrix = np.random.randint(0,3, size = (matrix_size,matrix_size))


def initialize_snake_gui():
    play_grid = (matrix_size*cell_size, matrix_size*cell_size)

    color_map = {0:(255,33,0), 1: (0,222,0),2: (0,0,222)} 


    matrix = np.random.randint(0,3, size = (matrix_size,matrix_size))
     


pygame.init()

pygame.display.set_caption('Quick Start')
window_surface = pygame.display.set_mode(play_grid)

manager = pygame_gui.UIManager(play_grid)

background = pygame.Surface(play_grid)
background.fill(pygame.Color(background_color))

clock = pygame.time.Clock()
is_running = True

while is_running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

    for row in range(matrix_size):
            for col in range(matrix_size):
                color = color_map[matrix[row][col]]
                pygame.draw.rect(window_surface, color, (col * cell_size, row * cell_size, cell_size, cell_size))

        # Update display
    pygame.display.flip()
    matrix = np.random.randint(0,3, size = (matrix_size,matrix_size))
    clock.tick(1)

# Quit pygame
pygame.quit()

def run():
    matrix_size = 10
    cell_size = 60

    play_grid = (matrix_size*cell_size, matrix_size*cell_size)

    background_color =(255,0,0)
    black = (0,0,0)

    color_map = {0:(255,33,0), 1: (0,222,0),2: (0,0,222)} 


    matrix = np.random.randint(0,3, size = (matrix_size,matrix_size))

    pygame.init()

    pygame.display.set_caption('Quick Start')
    window_surface = pygame.display.set_mode(play_grid)

    manager = pygame_gui.UIManager(play_grid)

    background = pygame.Surface(play_grid)
    background.fill(pygame.Color(background_color))

    clock = pygame.time.Clock()
    is_running = True

    while is_running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

        for row in range(matrix_size):
                for col in range(matrix_size):
                    color = color_map[matrix[row][col]]
                    pygame.draw.rect(window_surface, color, (col * cell_size, row * cell_size, cell_size, cell_size))

            # Update display
        pygame.display.flip()
        matrix = np.random.randint(0,3, size = (matrix_size,matrix_size))
        #clock.tick(0.00001)
        pygame.time.wait(1000)

    # Quit pygame
    pygame.quit()







