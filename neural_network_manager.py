import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import numpy as np
import random

with open('nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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


def do_bptt(NNr, NNd, NNp, episode_history, batch_size: int):
    unkown_magic = 5
    q = unkown_magic
    p = unkown_magic

    for m in range(batch_size):
        episode = random.choice(episode_history)
        state = []
        actions = []
        policies = []
        values = []
        rewards = []
        for ep in episode:
            state.append(ep[0])
            values.append(ep[1])
            policies.append(ep[2])
            actions.append(ep[3])
            rewards.append(ep[4])

        i = random.randint(0, len(state)-1)
        state = state[i]
        actions = actions[i]
        policies = torch.tensor(policies[i], dtype=torch.float32)
        values = torch.tensor(values[i], dtype=torch.float32)
        rewards = torch.tensor(rewards[i], dtype=torch.float32)
    
        abstract_state = NNr(state)
        next_abstract_state, predicted_reward = NNd(abstract_state, actions)
        predicted_policies, predicted_values = NNp(next_abstract_state)

        predictions = [predicted_policies, predicted_values, predicted_reward]
        true = [policies, values, rewards]

        #Can add momentum
        optimizerR = torch.optim.SGD(NNr.parameters(), lr=0.05)
        optimizerD = torch.optim.SGD(NNd.parameters(), lr=0.05)
        optimizerP = torch.optim.SGD(NNp.parameters(), lr=0.05) 
        
        optimizerR.zero_grad()
        optimizerD.zero_grad()
        optimizerP.zero_grad()

        loss_fn = nn.MSELoss()
        lossR = loss_fn(predicted_values, values)
        lossD = loss_fn(predicted_reward, rewards)
        lossP = loss_fn(predicted_policies, policies)

        lossR.backward(retain_graph=True)
        lossD.backward(retain_graph=True)
        lossP.backward()

        nn.utils.clip_grad_norm_(NNr.parameters(), 3)
        nn.utils.clip_grad_norm_(NNd.parameters(), 3)
        nn.utils.clip_grad_norm_(NNp.parameters(), 3) 

        optimizerR.step()
        optimizerD.step()
        optimizerP.step()

    return NNr, NNd, NNp

class RepresentationNetwork(nn.Module):
    """Representation Network (NNr) - Maps raw observations to abstract state representations."""
    def __init__(self, input_dim=config["game_size"]**2, layers=nn_config["representation"]["layers"], output_dim=nn_config["abstract_state_dim"], output_layer=nn_config["representation"]["output_layer"]):
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
        state=torch.tensor(state, dtype=torch.float32)
        state=state.view(-1)
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
        abstract_state=torch.tensor(abstract_state, dtype=torch.float32)
        abstract_state=abstract_state.view(-1)
        action=torch.tensor(action, dtype=torch.float32)
        action=action.view(-1)
        

        x = torch.cat([abstract_state, action], dim=-1)  # Concatenate state and action
        x = self.fc(x)
        next_state = x[:nn_config["abstract_state_dim"]]
        reward = x[nn_config["abstract_state_dim"]:]
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
        abstract_state=torch.tensor(abstract_state, dtype=torch.float32)
        x = self.shared_fc(abstract_state)
        
        # Policy output
        policy = self.policy_fc(x)
        policy = self.policy_activation(policy)
        
        # Value output
        value = self.value_fc(x)
        value = self.value_activation(value)
        
        return policy, value

