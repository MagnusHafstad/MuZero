import pytest
import numpy as np


import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from U_tree import *
from Game import Snake

def test_select_child_node():
    game = Snake(5)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    u_tree.root.add_child(game.board, u_tree.root, 10, 1, "playing")
    u_tree.root.add_child(game.board, u_tree.root, 0, 1, "dead")
    u_tree.root.add_child(game.board, u_tree.root, 0, 1, "dead")
    u_tree.root.add_child(game.board, u_tree.root, 0, 1, "dead")

    assert u_tree.select_child_node(u_tree.root) == u_tree.root.children[0]

    u_tree.root.children[0].visit_count = 10
    u_tree.root.children[1].visit_count = 2
    u_tree.root.children[2].visit_count = 1
    u_tree.root.children[3].visit_count = 0

    # Test that the node with the highest UCB is selected
    assert u_tree.select_child_node(u_tree.root) == u_tree.root.children[0]

    # Test that the node with the highest UCB is selected
    u_tree.root.children[0].visit_count = 0
    u_tree.root.children[1].visit_count = 0
    u_tree.root.children[2].visit_count = 0
    u_tree.root.children[3].visit_count = 0

    assert u_tree.select_child_node(u_tree.root) == u_tree.root.children[0]

    # Test that the node with the highest UCB is selected
    u_tree.root.children[0].visit_count = 0
    u_tree.root.children[1].visit_count = 0
    u_tree.root.children[2].visit_count = 0
    u_tree.root.children[3].visit_count = 1

    assert u_tree.select_child_node(u_tree.root) == u_tree.root.children[3]

    # Test that the node with the highest UCB is selected
    u_tree.root.children[0].visit_count = 0
    u_tree.root.children[1].visit_count = 0
    u_tree.root.children[2].visit_count = 1
    u_tree.root.children[3].visit_count = 0

    assert u_tree.select_child_node(u_tree.root) == u_tree.root.children[2]

def test_search_to_leaf():
    game = Snake(5)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    u_tree.root.add_child(game.board, u_tree.root, 10, "playing")    

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
    assert u_tree.root.children[0].reward == 0.001 # alive
    assert u_tree.root.children[1].reward == -1 # dead
    assert u_tree.root.children[0].parent == u_tree.root


def test_do_backpropagation():
    game = Snake(5)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    new_state, reward =game.simulate_game_step( game.board , 0)
    u_tree.root.add_child(new_state, u_tree.root,reward, game.status)
    new_state, reward =game.simulate_game_step( game.board , 0)
    u_tree.root.children[0].add_child(new_state, u_tree.root.children[0],reward, game.status)

    # Validity of test:
    assert u_tree.root.reward == 0.0
    assert u_tree.root.children[0].reward == 10.0
    assert u_tree.root.children[0].children[0].reward == 20.0

    # Test of do backpropagation

    u_tree.do_backpropagation(u_tree.root.children[0].children[0], 0.0, 0.9)

    # Visit counts should be updated
    assert u_tree.root.visit_count == 1
    assert u_tree.root.children[0].visit_count == 1
    assert u_tree.root.children[0].children[0].visit_count == 1

    # Rewards should be updated
    assert np.isclose(u_tree.root.children[0].children[0].reward, 20*0.9**0)
    assert np.isclose(u_tree.root.children[0].reward, 10 + 20 * 0.9**1)
    assert np.isclose(u_tree.root.reward, 10 * 0.9 + 20 * 0.9**2)
    
def test_do_rollout():
    #MCT_game.get_next_state_and_reward, MCT_game.get_policy
    game = Snake(5)
    np.random.seed(42)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    
    for action in u_tree.actions:
        game = Snake(5)
        new_state, reward =game.simulate_game_step( game.board ,action )
        u_tree.root.add_child(new_state, u_tree.root,reward, game.status)

    # Valitdity of test: 
    assert len(u_tree.root.children) == 4

    # Test of do rollout
    accum_reward = u_tree.do_rollout(u_tree.root.children[0], 2, game.get_policy, game.get_next_state_and_reward)
    
    assert len(accum_reward) == 3


def test_do_rollout_early_stop():
    bad_situation = np.array([[4, 3, 0, 0, 0],
                              [1, 2, 0, 0, 0],
                              [0, 0, -1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])
    game = Snake(5)
    game.board = bad_situation.copy()
    
    np.random.seed(42)
    u_tree = U_tree(game.board, 10, [0,1,2,3])
    
    for action in u_tree.actions:
        game = Snake(5)
        game.board = bad_situation.copy()
        new_state, reward =game.simulate_game_step( game.board ,action)
        u_tree.root.add_child(new_state, u_tree.root,reward, game.status)
    accum_reward = u_tree.do_rollout(u_tree.root.children[0], 2, game.get_policy, game.get_next_state_and_reward)
    
    assert len(accum_reward) == 1
    assert sum(accum_reward) == 30


