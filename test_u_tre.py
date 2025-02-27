from U_tree import U_tree
import torch
import torch.nn as nn
import numpy as np


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
    tree = U_tree([1,2,1,1,1], 10, action_list)
   
    for i in range(20):
        tree.MCTS(action_list, NNd, NNs, NNp)

    tree.print_tree(tree.root)

testTree()

policy = [0.7,0.1,0.1,0.1]
action_list = [1,2,3,4]
tree = U_tree([1,2,1,1,1], 10, action_list)

for i in range(10):
    print(tree.get_action(policy))

