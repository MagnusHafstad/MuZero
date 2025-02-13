import pygame
import pygame_gui
import numpy as np

from Game import Snake



class SnakeGUI:
    def __init__(self, matrix_size):
        pygame.init()
        self.matrix_size = matrix_size
        self.cell_size = 60
        self.play_grid = (matrix_size*self.cell_size, matrix_size*self.cell_size)
        self.snake_game = Snake(matrix_size)
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

    def game_loop(self):
        self.snake_game.get_next_location()
        self.snake_game.set_next_state()
        self.update_gui(self.snake_game.get_board())
              

    def run(self): 
        state = self.snake_game.get_board()

        clock = pygame.time.Clock()
        is_running = True

        while is_running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

            self.game_loop()

            clock.tick(3)

        # Quit pygame
        pygame.quit()




def game_loop(self) -> None:
        while self.state == "playing":
            self.get_next_location()
            self.set_next_state()
            print(self.board)
            #Update GUI
        print(self.state)


if __name__ == "__main__":
     snake = SnakeGUI(8)
     snake.run()