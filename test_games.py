import pytest
import Game
import numpy as np

def test_setup_game():
    snake_game = Game.Snake(5)
    assert np.array_equal(snake_game.board, [[0,0,0,0,0],
                                             [0,0,0,0,0],
                                             [1,0,2,0,0],
                                             [0,0,0,0,0],
                                             [0,0,0,0,0]])
    assert snake_game.direction == "right"


def test_representations():
    snake_game = Game.Snake(5)
    snake_game.board = np.array([[0,0,0,0,0],
                                 [1,1,1,0,0],
                                 [1,0,2,0,0],
                                 [0,0,0,0,0],
                                 [0,0,0,0,0]])
    snake_game.snake = [(2,0), (1,0), (1,1), (1,2)]
    old_board = snake_game.board.copy()
    old_snake = snake_game.snake.copy()
    nn_rep = snake_game.get_game_state()
    snake_game.board = None
    snake_game.snake = None

    snake_game.nn_state_to_game_state(nn_rep)
    assert np.array_equal(snake_game.board, old_board)
    assert np.array_equal(snake_game.snake, old_snake)

test_representations()