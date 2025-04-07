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

def do_bptt(NNr, NNd, NNp, episode_history, batch_size: int): #EP_hist: real_game_states[-1],root_value, final_policy_for_step, next_action, next_reward
    
    look_ahead = config["train_config"]["look_ahead"]
    look_back = config["train_config"]["look_back"]

    for i in range(batch_size):
        print("training: ", i)
    
        episode, episode_idx = get_random_instance_with_idx(episode_history)#[config["train_config"]["train_interval"]:]) kan kanskje begrene til å sepå de siste episodene
        count = 0
        while len(episode) <= 1:
            episode = random.choice(episode_history)
            count = count + 1
            if count == 10:
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


    # Predicted PVR
    state_tensor = torch.from_numpy(np.array(state)).float()
    state_tensor.requires_grad = True  # Ensure the tensor requires gradient
    abstract_state = NNr(state_tensor)
    print("abstract_state: ",abstract_state)
    next_abstract_state, predicted_reward, __ = NNd(abstract_state, actions)
    predicted_policies, predicted_values = NNp(next_abstract_state)
    print("Predicted values is leaf(after network): ", predicted_values.is_leaf)

        # predicted_reward = torch.tensor(predicted_reward, dtype=torch.float32)
        # predicted_policies =torch.tensor(predicted_policies,dtype=torch.float32)
        # predicted_values =torch.tensor(predicted_values,dtype=torch.float32)

    rewards = torch.tensor(rewards, dtype=torch.float32)
    policies =torch.tensor(policies,dtype=torch.float32)
    
    # values = [t.unsqueeze(0) for t in values]
    # values = torch.cat(values)
    values = torch.cat(values)

    predictions = [predicted_policies, predicted_values, predicted_reward]
    true = [policies, values, rewards]
    
    loss_fn = nn.MSELoss()

    predicted_values = predicted_values.squeeze(1)
    print("Predicted values is leaf (squeeze): ", predicted_values.is_leaf)
    
    # R NNr
    optimizerR = torch.optim.SGD(NNr.parameters(), lr=0.05)
    optimizerR.zero_grad()
    predicted_values = predicted_values.view(-1)
    values = values.view(-1)
    
    print("Shape is the same: ", predicted_values.shape == values.shape ,predicted_values.shape, values.shape)
    print("Requires grad check:", predicted_values.requires_grad, values.requires_grad)
    print("Grad_fn for predicted values: ", predicted_values.grad_fn)
    print("NNr output:", predicted_values)
    print("Predicted values is leaf: ", predicted_values.is_leaf)

    lossR = loss_fn(predicted_values, values)
    lossR.backward(retain_graph=True)
    print("Loss value:", lossR.item())

    torch.nn.utils.clip_grad_norm_(NNr.parameters(), 3)
    optimizerR.step()
    for param in NNr.parameters():
        print(param.grad is None, param.shape)
    for name, param in NNr.named_parameters():
        if param.grad is None:
            print(f"❌ No gradient for {name}")
        else:
            print(f"✅ Gradient found for {name}")
    print(f"Gradient NNr: {NNr.parameters().__next__().grad}")
    
    # D NNd
    optimizerD = torch.optim.SGD(NNd.parameters(), lr=0.05)
    optimizerD.zero_grad()
    lossD = loss_fn(predicted_reward, rewards)
    lossD.backward(retain_graph=True)
    optimizerD.step()
    torch.nn.utils.clip_grad_norm_(NNd.parameters(), 3)
    print(f"Gradient NNd: {NNd.parameters().__next__().grad}")

    # P NNp
    optimizerP = torch.optim.SGD(NNp.parameters(), lr=config["train_config"]["lr_NNp"])
    optimizerP.zero_grad()
    lossP = loss_fn(predicted_policies, policies)
    lossP.backward()

    torch.nn.utils.clip_grad_norm_(NNp.parameters(), 3)
    optimizerP.step()
        

    if config["train_config"]["train_verbal"] == True:
        print(f"Gradient NNr: {NNr.parameters().__next__().grad}")
        print(f"Gradient NNd: {NNd.parameters().__next__().grad}")
        print(f"Gradient NNp: {NNp.parameters().__next__().grad}")
        print(f"Loss R: {lossR.item()}, Loss D: {lossD.item()}, Loss P: {lossP.item()}")

    NNr.loss.append(lossR.item())
    NNd.loss.append(lossD.item())
    NNp.loss.append(lossP.item())
   
   
    return NNr, NNd, NNp


def do_bptt_second(NNr, NNd, NNp, episode_history, batch_size: int):

    look_ahead = config["train_config"]["look_ahead"]
    look_back = config["train_config"]["look_back"]
    print(f"Parameters NNr: {NNr.parameters().__next__()}")
    print(f"Parameters NNd: {NNd.parameters().__next__()}")
    print(f"Parameters NNp: {NNp.parameters().__next__()}")


    neural_networks = [NNr,NNd,NNp]
    for i in range(batch_size):

        episode, episode_idx = get_random_instance_with_idx(episode_history)#[config["train_config"]["train_interval"]:]) kan kanskje begrene til å sepå de siste episodene
        count = 0
        while len(episode) <= 1:
            episode = random.choice(episode_history)
            count = count + 1
            if count == 10:
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
            print(next_abstract_state.requires_grad)
            predicted_policy, predicted_value = NNp(next_abstract_state)
            predicted_policies.append(predicted_policy)
            predicted_rewards.append(predicted_reward)
            predicted_values.append(predicted_value)

        predicted_policies = torch.stack(predicted_policies)
        predicted_values = torch.stack(predicted_values)
        predicted_rewards = torch.stack(predicted_rewards)

        predicted = [predicted_policies, predicted_values, predicted_rewards]

        nn_param = list(NNr.parameters()) +list(NNd.parameters()) +list(NNp.parameters())
        optimizer = torch.optim.SGD(nn_param, lr = 0.0001)
        # optimizerR = torch.optim.SGD(NNr.parameters(), lr=0.0001)
        # optimizerD = torch.optim.SGD(NNd.parameters(), lr=0.0001)
        # optimizerP = torch.optim.SGD(NNp.parameters(), lr=0.0001)
        # optimizerR.zero_grad()
        # optimizerD.zero_grad()
        # optimizerP.zero_grad()
        optimizer.zero_grad()

        loss_fn = nn.MSELoss()


        lossP = loss_fn(predicted[0], targets[0])
        lossV= loss_fn(predicted[1], targets[1])
        lossR = loss_fn(predicted[2], targets[2])
        #loss1.backward(retain_graph=True)
        
        loss = lossP + lossV + lossR # + const*abs(param)**2
        loss.backward(retain_graph=True)
        optimizer.step()
        #loss3.backward(retain_graph=True)
    print("Loss value:", loss.item())
    if config["train_config"]["train_verbal"] == True:
        print(f"Gradient NNr: {NNr.parameters().__next__().grad}")
        print(f"Gradient NNd: {NNd.parameters().__next__().grad}")
        print(f"Gradient NNp: {NNp.parameters().__next__().grad}")

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
        #abstract_state=torch.tensor(abstract_state, dtype=torch.float32)
        
        #abstract_state=abstract_state.view(-1)
        action=torch.tensor(action, dtype=torch.float32)
        
        action = action.unsqueeze(1)

        #action=action.view(-1)
        

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

