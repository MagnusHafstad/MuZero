import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml

with open('nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class NeuralNetworkManager:
    """Manages the training and deployment of MuZero’s three neural networks."""
    def __init__(self, state_dim, abstract_state_dim, config):
        #  Initialize the three neural networks using the provided configuration
        self.representation_network = RepresentationNetwork(state_dim, abstract_state_dim, config["representation"]["layers"])
        self.dynamics_network = DynamicsNetwork(abstract_state_dim, 1, config["dynamic"]["layers"]) #the 1 is the size of the action input
        self.prediction_network = PredictionNetwork(abstract_state_dim, 1, config["prediction"]["layers"]) #the 1 is the size of the action input
        
        self.optimizer = optim.Adam(
            list(self.representation_network.parameters()) +
            list(self.dynamics_network.parameters()) +
            list(self.prediction_network.parameters()), learning_rate = config["learning_rate"])
    
    def train_step(self, batch):
        """Runs one step of training using backpropagation through time (BPTT)."""
        states, actions, policies, values, rewards = batch
        
        # Forward pass
        abstract_states = self.representation_network(states)
        pred_policies, pred_values = self.prediction_network(abstract_states)
        
        loss_policy = F.cross_entropy(pred_policies, policies)
        loss_value = F.mse_loss(pred_values.squeeze(), values)
        loss_reward = 0
        
        # Simulate dynamics
        for i in range(actions.shape[1]):  # Rollout through time
            abstract_states, pred_rewards = self.dynamics_network(abstract_states, actions[:, i])
            loss_reward += F.mse_loss(pred_rewards.squeeze(), rewards[:, i])
        
        loss = loss_policy + loss_value + loss_reward
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, state):
        """Deploys the trained networks to make predictions from a given game state."""
        with torch.no_grad():
            abstract_state = self.representation_network(state)
            policy, value = self.prediction_network(abstract_state)
        return policy, value

def pick_activation_func(name):
    """Picks an activation function based on its name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "linear":
        return nn.Identity()
    elif name == "softmax":
        return nn.Softmax()
    else:
        raise ValueError("Invalid activation function name.")


class RepresentationNetwork(nn.Module):
    """Representation Network (NNr) - Maps raw observations to abstract state representations."""
    def __init__(self, input_dim=nn_config["state_dim"], layers=nn_config["representation"]["layers"], output_dim=nn_config["abstract_state_dim"], output_layer=nn_config["representation"]["output_layer"]):
        super().__init__()
        layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            layers_list.append(pick_activation_func(hidden_dim["type"]))
            prev_dim = hidden_dim["dim"]
        
        layers_list.append(nn.Linear(prev_dim, output_dim))
        layers_list.append(pick_activation_func(output_layer))
        self.fc = nn.Sequential(*layers_list)
    
    def forward(self, state): 
        # Ensure state tensor has a batch dimension
        state=torch.tensor(state)
        state=state.view(-1)
        print(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        return self.fc(state)

class DynamicsNetwork(nn.Module):
    """Dynamics Network (NNd) - Predicts next state and reward given an abstract state and an action."""
    def __init__(self, action_dim=nn_config["action_dim"], abstract_state_dim=nn_config["abstract_state_dim"], layers=nn_config["dynamics"]["layers"], reward_dim=nn_config["reward_dim"], output_layer=nn_config["representation"]["output_layer"]):
        super().__init__()
        layers_list = []
        self.abstract_state_dim = abstract_state_dim
        prev_dim = action_dim + abstract_state_dim

        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            prev_dim = hidden_dim["dim"]
        
        layers_list.append(nn.Linear(prev_dim, abstract_state_dim + reward_dim))
        layers_list.append(pick_activation_func(output_layer))

        self.fc = nn.Sequential(*layers_list)
    
    def forward(self, abstract_state, action):
        # Ensure state and action tensors have a batch dimension
        abstract_state=torch.tensor(abstract_state)
        action=torch.tensor(action)
        action=action.view(-1)
        if abstract_state.dim() == 1:
            abstract_state = abstract_state.unsqueeze(0)  # Add batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Add batch dimension

        x = torch.cat([abstract_state, action], dim=-1)  # Concatenate state and action
        x = self.fc(x)
        next_state = x[:, :nn_config["abstract_state_dim"]]
        reward = x[:, nn_config["abstract_state_dim"]:]
        return next_state, reward

class PredictionNetwork(nn.Module):
    """Prediction Network (NNp) - Outputs policy and value estimates from an abstract state."""
    def __init__(self, abstract_state_dim=nn_config["abstract_state_dim"], policy_dim=nn_config["policy_dim"], layers=nn_config["prediction"]["layers"], reward_dim=nn_config["reward_dim"], policy_output_layer=nn_config["prediction"]["policy_output_layer"], reward_output_layer=nn_config["prediction"]["reward_output_layer"]):
        super().__init__()
        
        # Shared layers
        layers_list = []
        prev_dim = abstract_state_dim
        
        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            layers_list.append(pick_activation_func(hidden_dim["type"]))
            prev_dim = hidden_dim["dim"]
        
        self.shared_fc = nn.Sequential(*layers_list)
        
        # Policy output layers
        self.policy_fc = nn.Linear(prev_dim, policy_dim)
        self.policy_activation = pick_activation_func(policy_output_layer)
        
        # Value output layers
        self.value_fc = nn.Linear(prev_dim, reward_dim)
        self.value_activation = pick_activation_func(reward_output_layer)
    
    def forward(self, abstract_state):
        abstract_state=torch.tensor(abstract_state)
        x = self.shared_fc(abstract_state)
        
        # Policy output
        policy = self.policy_fc(x)
        policy = self.policy_activation(policy)
        
        # Value output
        value = self.value_fc(x)
        value = self.value_activation(value)
        
        return policy, value

