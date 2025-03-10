from U_tree import U_tree
import torch
import torch.nn as nn
import numpy as np
import Game


from neural_network_manager import *
import torch

NNr = RepresentationNetwork()
if nn_config["representation"]["load"]:
    NNr.load_state_dict(torch.load('NNr.pth'))
NNd = DynamicsNetwork()
if nn_config["dynamics"]["load"]:
    NNd.load_state_dict(torch.load('NNd.pth'))
NNp = PredictionNetwork()
if nn_config["prediction"]["load"]:
    NNp.load_state_dict(torch.load('NNp.pth'))



def get_policy(node, snake_game):
    policy = [0.25,0.25,0.25,0.25]
    state_reward = snake_game.get_next_state_and_reward(node)[1]
    return policy, state_reward


def testTree():
    #Generate tree:
    snake_game = Game.Snake(5)

    # snake_game.board = np.array([[0,0,0,0,0],
    #                              [0,0,0,0,0],
    #                              [1,0,2,0,0],
    #                              [0,0,0,0,0],
    #                              [0,0,0,0,0]])
    # snake_game.snake = snake_game.snake = [(2,0), (1,0), (1,1), (1,2)]
    
    nn_rep = snake_game.get_real_nn_game_state()

    abs_state = NNr.forward(nn_rep)

    action_list = [0,1,2,3]
    tree = U_tree(snake_game.get_real_nn_game_state(), 10, action_list, snake_game)
    

    for i in range(10):
        tree.MCTS(snake_game.simulate_game_step, get_policy)

    tree.print_tree(tree.root)





testTree()

