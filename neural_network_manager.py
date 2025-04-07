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

def do_bptt_second(NNr, NNd, NNp, episode_history, batch_size: int):

    look_ahead = config["train_config"]["look_ahead"]
    look_back = config["train_config"]["look_back"]

    nn_param = list(NNr.parameters()) +list(NNd.parameters()) +list(NNp.parameters())
    optimizer = torch.optim.SGD(nn_param, lr = 0.001)
    for i in range(batch_size):
        optimizer.zero_grad()
        episode, episode_idx = get_random_instance_with_idx(episode_history)#[config["train_config"]["train_interval"]:]) kan kanskje begrene til å sepå de siste episodene
        count = 0
        while len(episode) <= 2:
            episode = random.choice(episode_history)
            count = count + 1
            if count > 10:
                print("I give up, I deserve nothing but death")

        step, step_idx = get_random_instance_with_idx(episode)


        if step_idx <= look_back:
                look_back = step_idx
        if len(episode) - step_idx <= look_ahead:
            look_ahead = len(episode) - step_idx - 1
            
        step_array = episode[step_idx - look_back : step_idx + look_ahead + 1]

        state = []
        actions = []
        policies = []
        values = []
        rewards = []
        for step in step_array:
            state.append(step[0].flatten())
            values.append(step[1])
            policies.append(step[2])
            actions.append(step[3])
            rewards.append(step[4])

        look_back_states = state[0:look_back + 1]
        look_ahead_actions = actions[look_back:]

        policies = torch.tensor(policies[look_back:],dtype=torch.float32)
        values = torch.tensor(values[look_back:],dtype=torch.float32)
        rewards = torch.tensor(rewards[look_back:],dtype=torch.float32)
        
        targets =[policies, values, rewards]

        state_tensor = torch.from_numpy(np.array(look_back_states[0])).float()
        state_tensor.requires_grad = True  # Ensure the tensor requires gradient
        abstract_state = NNr(state_tensor)

        #abstract_state = NNr(look_back_states[0])

        predicted_policies = []
        predicted_rewards = []
        predicted_values = []

        for action in look_ahead_actions:
            next_abstract_state, predicted_reward, __ = NNd(abstract_state, [action])
            predicted_policy, predicted_value = NNp(next_abstract_state)
            predicted_policies.append(predicted_policy)
            predicted_rewards.append(predicted_reward)
            predicted_values.append(predicted_value)
            abstract_state = next_abstract_state

        predicted_policies = torch.stack(predicted_policies)
        predicted_values = torch.stack(predicted_values)
        predicted_rewards = torch.stack(predicted_rewards)

        print("Predicted values stats:", predicted_values.min(), predicted_values.max(), len(predicted_values))
        print("real values: ", values.min(), values.max(), len(values))


        predicted = [predicted_policies, predicted_values, predicted_rewards]

        loss_fn = nn.MSELoss()
        lossP = loss_fn(predicted[0], targets[0])
        lossV= loss_fn(predicted[1], targets[1])
        lossR = loss_fn(predicted[2], targets[2])
        regularization_term = sum(torch.sum(param ** 2) for param in nn_param)
        loss = lossP + lossV + lossR + 0.01 * regularization_term

        print("LossP:", lossP.item(), "LossV:", lossV.item(), "LossR:", lossR.item())

        loss.backward(retain_graph=False)
        optimizer.step()
        print("-------------------------------------")

    print("Loss value:", loss.item())
    if config["train_config"]["train_verbal"] == True:
        with open('gradient_output.txt', 'w') as f:  # Open file in write mode
            f.write(f"Gradient NNr: {NNr.parameters().__next__().grad}\n")
            f.write(f"Gradient NNd: {NNd.parameters().__next__().grad}\n")
            f.write(f"Gradient NNp: {NNp.parameters().__next__().grad}\n")

    return loss.item()
    


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
    def __init__(self, action_dim=nn_config["action_dim"], abstract_state_dim=nn_config["abstract_state_dim"], layers=nn_config["dynamics"]["layers"], reward_dim=nn_config["reward_dim"], state_output_layer=nn_config["dynamics"]["state_output_layer"], reward_output_layer=nn_config["dynamics"]["reward_output_layer"]):
        super().__init__()
        layers_list = []
        self.abstract_state_dim = abstract_state_dim
        prev_dim = action_dim + abstract_state_dim
        self.loss = []

        for hidden_dim in layers:
            layers_list.append(nn.Linear(prev_dim, hidden_dim["dim"]))
            layers_list.append(pick_activation_func(hidden_dim["type"]))
            prev_dim = hidden_dim["dim"]
        
        self.fc = nn.Sequential(*layers_list)
        self.state_output_activation = pick_activation_func(state_output_layer)
        self.reward_output_activation = pick_activation_func(reward_output_layer)
        self.state_output_layer = nn.Linear(prev_dim, abstract_state_dim)
        self.reward_output_layer = nn.Linear(prev_dim, reward_dim)
        
    
    def forward(self, abstract_state, action):
        action=torch.tensor(action, dtype=torch.float32)
        action = action.unsqueeze(1)        
        x = torch.cat([abstract_state, action], dim=-1)  # Concatenate state and action
        x = self.fc(x)
        

        next_state = self.state_output_layer(x)
        next_state = self.state_output_activation(next_state)
        reward = self.reward_output_layer(x)
        reward = self.reward_output_activation(reward)
        
        return next_state, reward, "playing"

class PredictionNetwork(nn.Module):
    """Prediction Network (NNp) - Outputs policy and value estimates from an abstract state."""
    def __init__(self, abstract_state_dim=nn_config["abstract_state_dim"], policy_dim=nn_config["policy_dim"], layers=nn_config["prediction"]["layers"], reward_dim=nn_config["reward_dim"], policy_output_layer=nn_config["prediction"]["policy_output_layer"], value_output_layer=nn_config["prediction"]["value_output_layer"]):
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
        self.value_activation = pick_activation_func(value_output_layer)
    
    def forward(self, abstract_state):
        x = self.shared_fc(abstract_state)
        
        # Policy output
        policy = self.policy_fc(x)
        policy = self.policy_activation(policy)
        
        # Value output
        value = self.value_fc(x)
        value = self.value_activation(value)
        
        return policy, value

