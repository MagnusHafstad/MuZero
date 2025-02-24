import neural_network_manager
import torch

state_dim = 20
abstract_state_dim = 5
hidden_layers = [20, 20, 20]
activation_func = torch.nn.ReLU()
action_dim = 1


representation_network = neural_network_manager.RepresentationNetwork(state_dim, abstract_state_dim, hidden_layers, activation_func)
dynamics_network = neural_network_manager.DynamicsNetwork(abstract_state_dim, action_dim, hidden_layers, activation_func)
prediction_network = neural_network_manager.PredictionNetwork(abstract_state_dim, action_dim, hidden_layers, activation_func)


