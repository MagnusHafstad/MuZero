from U_tree import U_tree
import torch
import torch.nn as nn
import numpy as np
import Game
import copy
import tqdm

from neural_network_manager import *
import torch

# NNr = RepresentationNetwork()
# if nn_config["representation"]["load"]:
#     NNr.load_state_dict(torch.load('NNr.pth'))
# NNd = DynamicsNetwork()
# if nn_config["dynamics"]["load"]:
#     NNd.load_state_dict(torch.load('NNd.pth'))
# NNp = PredictionNetwork()
# if nn_config["prediction"]["load"]:
#     NNp.load_state_dict(torch.load('NNp.pth'))






def testTree():
    #Generate tree:
    snake_game = Game.Snake(5)

    # snake_game.board = np.array([[0,0,0,0,0],
    #                              [0,0,0,0,0],
    #                              [1,0,2,0,0],
    #                              [0,0,0,0,0],
    #                              [0,0,0,0,0]])
    # snake_game.snake = snake_game.snake = [(2,0), (1,0), (1,1), (1,2)]
    
    action_list = [0,1,2,3]
    MCT_game = snake_game.copy()
    tree = U_tree(MCT_game.board, 50, action_list)

    loopnr = 0
    while snake_game.status == "playing":
        MCT_game = snake_game.copy()
        tree = U_tree(MCT_game.board, 1, action_list)
        tree.print_tree(tree.root)
        #print(snake_game.board)
        
        for i in tqdm.tqdm(range(500)):
            tree.MCTS(MCT_game.get_next_state_and_reward, MCT_game.get_policy)
        #tree.print_tree(tree.root)
        snake_game.direction = tree.get_action(tree.normalize_visits())
        snake_game.get_next_location()
        snake_game.set_next_state()
        print(snake_game.board, loopnr)

        if config.get('head'):
            snake_game.gui.update_gui(snake_game.board)

        loopnr += 1
    #tree.print_tree(tree.root)



testTree()


