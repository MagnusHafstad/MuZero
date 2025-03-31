import pytest
import numpy as np


import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Game import Snake


def test_setup_game():
    snake_game = Snake(5)
    assert np.array_equal(snake_game.board, [[0,0,0,0,0],
                                            [0,0,0,0,0],
                                            [1,2,0,0,-1],
                                            [0,0,0,0,0],
                                            [0,0,0,0,0]])
    assert snake_game.direction == 0


def test_simulate_game_step():
    board = np.array([[ 0,  0,  0, -1,  0],
                      [ 0,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  0],
                      [ 2,  1,  0,  0,  0],
                      [ 3,  0,  0,  0,  0]])
    game = Snake(5)
    game.board = board.copy()

    new_board, reward = game.simulate_game_step(board,0)

    assert np.array_equal(new_board, [[ 0,  0,  0, -1,  0],
                                      [ 0,  0,  0,  0,  0],
                                      [ 0,  0,  0,  0,  0],
                                      [ 1,  0,  0,  0,  0],
                                      [ 2,  3,  0,  0,  0]])
    assert reward == 0.001
