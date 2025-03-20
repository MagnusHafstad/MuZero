import pytest
import numpy as np


import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from U_tree import *
from Game import Snake



def test_add_child():
    game = Snake(5)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    for action in u_tree.actions:
        game = Snake(5)
        new_state, reward =game.simulate_game_step( game.board ,action )
        u_tree.root.add_child(new_state, u_tree.root,reward, game.status)

    assert len(u_tree.root.children) == 4
    assert u_tree.root.children[0].depth == 1
    assert u_tree.root.children[0].status == "playing"
    assert u_tree.root.children[0].reward == 10 # alive
    assert u_tree.root.children[1].reward == 0 # dead
    assert u_tree.root.children[0].parent == u_tree.root

    