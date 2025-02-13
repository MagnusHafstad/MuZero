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