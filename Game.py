import pygame
import numpy as np
import snake_gui
import yaml

def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('./config.yaml')


def get_game():
    match config.get('game'):
        case "snake":
            return Snake(config.get('game_size'))
    raise ValueError("Invalid game type in config.yaml")

class Snake():
    def __init__(self, size, head=True):
        # 0 is empty, 1 is snake, 2 is food
        food = 2
        snake = 1
        self.board = np.zeros((size, size), dtype=int)
        self.direction = 0 #right (see get next location)
        self.snake = [(len(self.board)//2, 0)]
        self.board[self.snake[0][0], self.snake[0][1]] = snake
        self.board[(len(self.board)//2,len(self.board)//2)] = food  
        self.status = "playing"
        self.head = head
        if self.head:
            self.gui = snake_gui.SnakeGUI(size)
            self.clock = pygame.time.Clock()

    
    def place_food(self) -> None:
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            print("Game Over")
        random_empty_cell = empty_cells[np.random.choice(empty_cells.shape[0])]
        self.board[random_empty_cell[0], random_empty_cell[1]] = 2
    

    def get_next_location(self) -> tuple:
        """
        0: right
        1: left
        2: up
        3: down
        """
        if self.direction == 0:
            return (self.snake[-1][0], self.snake[-1][1] + 1)
        elif self.direction == 1:
            return (self.snake[-1][0], self.snake[-1][1] - 1)
        elif self.direction == 2:
            return (self.snake[-1][0] - 1, self.snake[-1][1])
        elif self.direction == 3:
            return (self.snake[-1][0] + 1, self.snake[-1][1])
        
    def set_next_state(self) -> None:
        next_location = self.get_next_location()
        
        if next_location[0] > len(self.board)-1 or next_location[1] > len(self.board)-1:
            self.status = "game_over"
        elif next_location[0] < 0 or next_location[1] < 0:
            self.status = "game_over"
        elif self.board[next_location[0], next_location[1]] == 1:
            self.status = "game_over"

        
        elif self.board[next_location[0], next_location[1]] == 2:
            self.snake.append(next_location)
            self.board[next_location[0], next_location[1]] = 1
            self.place_food()
        elif self.board[next_location[0], next_location[1]] == 0:
            self.board[self.snake[0][0],self.snake[0][1]] = 0
            self.snake = self.snake[1:] 
            self.snake.append(next_location)
            self.board[next_location[0], next_location[1]] = 1
        elif self.board.all() == 1:
            self.status = "Win"
    
    def get_board(self) -> np.array:
        return self.board
    
    def get_state(self) -> str:
        return self.board, self.snake

    def get_direction(self) -> str:
        return self.direction
    
    def set_direction(self, direction: str) -> None:
        self.direction = direction

    def game_loop(self) -> None:
        while self.status == "playing":
            if self.head:
                self.clock.tick(3)
                self.direction = self.gui.user_input(self.direction)
            self.get_next_location()
            self.set_next_state()
            if self.head:
                self.gui.update_gui(self.board)
            #Update GUI
        print(self.status)

    def get_real_nn_game_state(self):
        nn_board = np.copy(self.board)
        nn_board[nn_board == 2] = -1
        nn_board[nn_board == 1] = 0
        # TODO: Check if it is easier for the network if you reverse the snake list so the head is always at index 0
        for segment in range(len(self.snake)):
            nn_board[self.snake[segment][0], self.snake[segment][1]] = segment + 1
        return nn_board
    
    def nn_state_to_game_state(self, nn_game_state):
        snake = []
        for i in range(1, len(nn_game_state)): # If you run into index errors with more than 4 snake segments check this line try config.get('game_size') instead of len(nn_game_state)
            snake.append(np.argwhere(nn_game_state == i))
            if len(snake[-1]) == 0: # This is a very bad solution, but it works for now
                snake.pop()
        nn_game_state[nn_game_state > 0] = 1
        nn_game_state[nn_game_state == -1] = 2

        self.snake = [(pos[0][0], pos[0][1]) for pos in snake]
        self.board = nn_game_state

    def simulate_game_step(self, real_game_state, direction: str):
        """Simulates one gamestep and returns the next state and reward
           
           In the general case, the direction is the action taken by the agent"""
        self.nn_state_to_game_state(real_game_state)
        self.direction = direction
        self.get_next_location()
        self.set_next_state()
        print(self.board)
        if config.get('head'):
            self.gui.update_gui(self.board)
        if self.status == "game_over":
            return self.get_real_nn_game_state(), -10 + len(self.snake)
        elif self.status == "Win":
            return self.get_real_nn_game_state(), 10 + len(self.snake)
        
        return self.get_real_nn_game_state(), len(self.snake)





