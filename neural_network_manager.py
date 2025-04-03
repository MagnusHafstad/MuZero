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


def get_random_instance_with_idx(list:list):
    i = random.choice(range(len(list)))
    return list[i], i

def do_bptt(NNr, NNd, NNp, episode_history, batch_size: int):
    look_ahead = config["train_config"]["look_ahead"]
    look_back = config["train_config"]["look_back"]
    loss_fn = nn.MSELoss()

    # Define optimizers jointly for all networks
    parameters = list(NNr.parameters()) + list(NNd.parameters()) + list(NNp.parameters())
    optimizer = optim.SGD(parameters, lr=config["train_config"]["lr"], momentum=0.9)

    for i in range(batch_size):
        episode, _ = get_random_instance_with_idx(episode_history)
        while len(episode) <= 1:
            episode, _ = get_random_instance_with_idx(episode_history)

        step, step_idx = get_random_instance_with_idx(episode)

        actual_look_back = min(look_back, step_idx)
        actual_look_ahead = min(look_ahead, len(episode) - step_idx - 1)
        step_array = episode[step_idx - actual_look_back : step_idx + actual_look_ahead + 1]

        states = torch.stack([torch.from_numpy(step[0]).float().flatten() for step in step_array[:actual_look_back+1]])
        actions = [step[3] for step in step_array[actual_look_back:]]
        target_policies = torch.stack([torch.tensor(step[2]).float() for step in step_array[actual_look_back:]])
        target_values = torch.stack([torch.tensor(step[1]).float() for step in step_array[actual_look_back:]])
        target_rewards = torch.stack([torch.tensor(step[4]).float() for step in step_array[actual_look_back:]])

        # Forward pass
        hidden_state = NNr(states)
        predicted_policies = []
        predicted_values = []
        predicted_rewards = []

        for action in actions:
            
            action_tensor = torch.tensor([[action]], dtype=torch.float32)  # adds batch dimension explicitly
            hidden_state, reward = NNd(hidden_state, action_tensor)
            policy, value = NNp(hidden_state)
            predicted_rewards.append(reward)
            predicted_policies.append(policy)
            predicted_values.append(value)

        predicted_rewards = torch.stack(predicted_rewards)
        predicted_policies = torch.stack(predicted_policies)
        predicted_values = torch.stack(predicted_values)

        # Compute losses
        loss_reward = loss_fn(predicted_rewards, target_rewards)
        loss_policy = loss_fn(predicted_policies, target_policies)
        loss_value = loss_fn(predicted_values.squeeze(-1), target_values)

        # Total loss
        loss = loss_reward + loss_policy + loss_value

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=3)
        optimizer.step()

        # Store losses
        NNr.loss.append(loss.item())
        NNd.loss.append(loss.item())
        NNp.loss.append(loss.item())

        if config["train_config"].get("train_verbal", False):
            print(f"Batch {i+1}/{batch_size}: Loss = {loss.item():.4f} (Reward: {loss_reward.item():.4f}, Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})")

    return NNr, NNd, NNp

    


class RepresentationNetwork(nn.Module):
    """Representation Network (NNr) - Maps raw observations to abstract state representations."""
    def __init__(self, input_dim=config["game_size"]**2, layers=nn_config["representation"]["layers"], output_dim=nn_config["abstract_state_dim"], output_layer=nn_config["representation"]["output_layer"]):
        super().__init__()
        layers_list = []
        prev_dim = input_dim
        self.loss = []
        
        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            layers_list.append(pick_activation_func(hidden_dim["type"]))
            prev_dim = hidden_dim["dim"]
        
        layers_list.append(nn.Linear(prev_dim, output_dim))
        layers_list.append(pick_activation_func(output_layer))
        self.fc = nn.Sequential(*layers_list)
    
    def forward(self, state): 
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if necessary
        return self.fc(state)

class DynamicsNetwork(nn.Module):
    """Dynamics Network (NNd) - Predicts next state and reward given an abstract state and an action."""
    def __init__(self, action_dim=nn_config["action_dim"], abstract_state_dim=nn_config["abstract_state_dim"], layers=nn_config["dynamics"]["layers"], reward_dim=nn_config["reward_dim"], output_layer=nn_config["representation"]["output_layer"]):
        super().__init__()
        layers_list = []
        self.abstract_state_dim = abstract_state_dim
        prev_dim = action_dim + abstract_state_dim
        self.loss = []

        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            prev_dim = hidden_dim["dim"]
        
        layers_list.append(nn.Linear(prev_dim, abstract_state_dim + reward_dim))
        layers_list.append(pick_activation_func(output_layer))

        self.fc = nn.Sequential(*layers_list)
    
    def forward(self, abstract_state, action):
        # Ensure state and action tensors have a batch dimension
        abstract_state=torch.tensor(abstract_state, dtype=torch.float32)
        print("action", action)
        #abstract_state=abstract_state.view(-1)
        action=torch.tensor(action, dtype=torch.float32)
        print("action Tensor", action)
        action = action.unsqueeze(1)

        #action=action.view(-1)
        
        print("absState", abstract_state)
        x = torch.cat([abstract_state, action], dim=-1)  # Concatenate state and action
        x = self.fc(x)
  
        next_state = x[:,:nn_config["abstract_state_dim"]]
        reward = x[:,nn_config["abstract_state_dim"]:]
        
        return next_state, reward, "playing"

class PredictionNetwork(nn.Module):
    """Prediction Network (NNp) - Outputs policy and value estimates from an abstract state."""
    def __init__(self, abstract_state_dim=nn_config["abstract_state_dim"], policy_dim=nn_config["policy_dim"], layers=nn_config["prediction"]["layers"], reward_dim=nn_config["reward_dim"], policy_output_layer=nn_config["prediction"]["policy_output_layer"], reward_output_layer=nn_config["prediction"]["reward_output_layer"]):
        super().__init__()
        
        # Shared layers
        layers_list = []
        prev_dim = abstract_state_dim
        self.loss = []
        
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

