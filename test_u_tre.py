from U_tree import U_tree
import torch
import torch.nn as nn


import neural_network_manager
import torch

state_dim = 3
abstract_state_dim = 5
hidden_layers = [20, 20, 20]
activation_func = torch.nn.ReLU
action_dim = 1


NNs = neural_network_manager.RepresentationNetwork(state_dim, abstract_state_dim, hidden_layers, activation_func)
NNd= neural_network_manager.DynamicsNetwork(abstract_state_dim, action_dim, hidden_layers, activation_func)
NNp = neural_network_manager.PredictionNetwork(abstract_state_dim, action_dim, hidden_layers, activation_func)



def testTree():
    #Generate tree:
    action_list = [1,2,3,4]
    tree = U_tree([1,2,1,1,1], 3, action_list)
   

    tree.MCTS(action_list, NNd, NNs, NNp)

    tree.print_tree(tree.root)

testTree()