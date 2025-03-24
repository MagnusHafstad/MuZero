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

    action_list = [0,1,2,3]
    loopnr = 0
    while snake_game.status == "playing":
        MCT_game = snake_game.copy() 
        tree = U_tree(MCT_game.board, 30, action_list)
        #tree.print_tree(tree.root)
        #print(snake_game.board)
        
        for i in tqdm.tqdm(range(300)):
            tree.MCTS(MCT_game.get_next_state_and_reward, MCT_game.get_policy)
        #tree.print_tree(tree.root, 3)
        snake_game.direction = tree.get_final_action(tree.normalize_visits())
        snake_game.get_next_location()
        snake_game.set_next_state()
        print(snake_game.board, loopnr)

        if config.get('head'):
            snake_game.gui.update_gui(snake_game.board)

        loopnr += 1
    tree.save_tree()
    
    #tree.print_tree(tree.root)



testTree()


