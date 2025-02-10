import pygame
import numpy as np

class Snake():
    def __init__(self, size):
        # 0 is empty, 1 is snake, 2 is food
        food = 2
        snake = 1
        self.board = np.zeros((size, size), dtype=int)
        self.direction = "right"
        self.snake = [(len(self.board)//2, 0)]
        self.board[self.snake[0][0], self.snake[0][1]] = snake
        self.board[(len(self.board)//2,len(self.board)//2)] = food  
        self.state = "playing"

        
    
    def place_food(self) -> None:
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            print("Game Over")
        random_empty_cell = empty_cells[np.random.choice(empty_cells.shape[0])]
        self.board[random_empty_cell[0], random_empty_cell[1]] = 2
    

    def get_next_location(self) -> tuple:
        if self.direction == "right":
            return (self.snake[-1][0], self.snake[-1][1] + 1)
        elif self.direction == "left":
            return (self.snake[-1][0], self.snake[-1][1] - 1)
        elif self.direction == "up":
            return (self.snake[-1][0] - 1, self.snake[-1][1])
        elif self.direction == "down":
            return (self.snake[-1][0] + 1, self.snake[-1][1])
        
    def set_next_state(self) -> None:
        next_location = self.get_next_location()
        
        if next_location[0] > len(self.board)-1 or next_location[1] > len(self.board)-1:
            self.state = "game_over"
        elif self.board[next_location[0], next_location[1]] == 1:
            self.state = "game_over"

        
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
            self.state = "Win"
    
    def get_board(self) -> np.array:
        return self.board

    def game_loop(self) -> None:
        while self.state == "playing":
            self.get_next_location()
            self.set_next_state()
            print(self.board)
            #Update GUI
        print(self.state)


        

        
snake = Snake(5)
snake.game_loop()