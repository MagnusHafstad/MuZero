import pygame
import numpy as np
import snake_gui
import yaml
import copy
import numpy as np

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
        food = -1
        snake = 1
        self.board = np.zeros((size, size), dtype=int)
        self.direction = 0 #right (see get next location)
        self.board[len(self.board)//2, 0] = snake
        self.board[(len(self.board)//2,len(self.board)//2)] = food  
        self.status = "playing"
        self.head = head
        if self.head:
            self.gui = snake_gui.SnakeGUI(size)
            self.clock = pygame.time.Clock()

    def copy(self):
        """
        Returns a deep copy of the current game state
        """
        new_snake = Snake(len(self.board), self.head)
        new_snake.board = np.copy(self.board)
        new_snake.direction = self.direction
        new_snake.status = self.status
        if self.head:
            new_snake.gui = self.gui
            new_snake.clock = self.clock
        return new_snake
    
    def place_food(self) -> None:
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            print("Game Over")
        random_empty_cell = empty_cells[np.random.choice(empty_cells.shape[0])]
        self.board[random_empty_cell[0], random_empty_cell[1]] = -1
    

    def get_next_location(self) -> tuple:
        """
        Pure. Calclulates the next location for the snake head based on the current direction
        0: right
        1: left
        2: up
        3: down
        """
        current_head = np.unravel_index(np.argmax(self.board, axis=None), self.board.shape)
        if self.direction == 0:
            return (current_head[0], current_head[1] + 1)
        elif self.direction == 1:
            return (current_head[0], current_head[1] - 1)
        elif self.direction == 2:
            return (current_head[0] - 1, current_head[1])
        elif self.direction == 3:
            return (current_head[0] + 1, current_head[1])
        
    def get_game_status(self, next_location) -> bool:
        """
        Pure. Determines the game status based on the next location of the snake
        """
        if next_location[0] > len(self.board)-1 or next_location[1] > len(self.board)-1:
            return "game_over"
        elif next_location[0] < 0 or next_location[1] < 0:
            return "game_over"
        elif self.board[next_location[0], next_location[1]] > 0: # snake crashes into itself
            return "game_over"
        elif self.board.all() > 0:
            return "Win"
        return "playing"
    
    def handle_movement(self, next_location) -> None:
        """s
        Handles the movement of the snake based on the next location
        """
        snake_head = np.max(self.board)
        if self.board[next_location[0], next_location[1]] == -1:
            self.board[next_location[0], next_location[1]] = snake_head +1
            self.place_food()
        elif self.board[next_location[0], next_location[1]] == 0:
            self.board[next_location[0], next_location[1]] = snake_head +1
            self.board[self.board > 0] -= 1
        
    def set_next_state(self) -> None:
        """
        An in place function that updates the game state to the next game step
        """
        next_location = self.get_next_location()
        self.status = self.get_game_status(next_location)
        if self.status == "playing":
            self.handle_movement(next_location)
        
        

    def get_next_state_and_reward(self,state,  action: int) -> tuple:
        """
        Pure. Returns the next state and reward based on the action taken
        """
        temp_game = self.copy()
        temp_game.board = np.copy(state)
        temp_game.direction = action[0]
        temp_game.set_next_state()
        return temp_game.board, temp_game.calculate_reward(temp_game.status, np.max(temp_game.board)), temp_game.status
    
    def get_policy(self, node):
        policy = [0.25,0.25,0.25,0.25]
        state_value = self.calculate_reward(node.status, np.max(self.board))
        return policy, state_value
        

    def calculate_reward(self, status, len_snake) -> int:
        """based on a game state"""
        if status == "game_over":
            return -10 + len_snake
        elif status == "Win":
            return 10 + len_snake
        return len_snake

    def get_board(self) -> np.array:
        return self.board

    def get_direction(self) -> str:
        return self.direction
    
    def set_direction(self, direction: str) -> None:
        self.direction = direction

    def game_loop(self) -> None:
        """
        Starts the game in a human player mode
        """
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

    def simulate_game_step(self, real_game_state, direction: str):
        """
        Simulates one gamestep and returns the next state and reward
           
        In the general case, the direction is the action taken by the agent
        """
        self.direction = direction
        self.get_next_location()
        self.set_next_state()
        print(self.board)
        if config.get('head'):
            self.gui.update_gui(self.board)
        
        return self.board, self.calculate_reward(self.status, np.max(self.board))
    

    
