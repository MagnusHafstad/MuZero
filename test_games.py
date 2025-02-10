import pytest
import Game

def test_setup_game():
    snake_game = Game.Snake(5)
    assert snake_game.board == [[0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0]]
    assert snake_game.direction == "right"