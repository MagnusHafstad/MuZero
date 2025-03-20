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
                                            [1,0,-1,0,0],
                                            [0,0,0,0,0],
                                            [0,0,0,0,0]])
    assert snake_game.direction == 0



   