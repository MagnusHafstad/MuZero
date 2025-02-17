import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    """Representation Network (NNr) - Maps raw observations to latent state representations."""
    def __init__(self, input_dim, latent_dim, hidden_layers, activation_func):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_func())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

class DynamicsNetwork(nn.Module):
    """Dynamics Network (NNd) - Predicts next state and reward given a latent state and action."""
    def __init__(self, latent_dim, action_dim, hidden_layers, activation_func):
        super().__init__()
        layers_state = []
        layers_reward = []
        prev_dim = latent_dim + action_dim
        
        for hidden_dim in hidden_layers:
            layers_state.append(nn.Linear(prev_dim, hidden_dim))
            layers_state.append(activation_func())
            layers_reward.append(nn.Linear(prev_dim, hidden_dim))
            layers_reward.append(activation_func())
            prev_dim = hidden_dim
        
        layers_state.append(nn.Linear(prev_dim, latent_dim))
        layers_reward.append(nn.Linear(prev_dim, 1))
        
        self.fc_state = nn.Sequential(*layers_state)
        self.fc_reward = nn.Sequential(*layers_reward)
    
    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        next_state = self.fc_state(x)
        reward = self.fc_reward(x)
        return next_state, reward

class PredictionNetwork(nn.Module):
    """Prediction Network (NNp) - Outputs policy and value estimates from a latent state."""
    def __init__(self, latent_dim, action_dim, hidden_layers, activation_func):
        super().__init__()
        layers_policy = []
        layers_value = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_layers:
            layers_policy.append(nn.Linear(prev_dim, hidden_dim))
            layers_policy.append(activation_func())
            layers_value.append(nn.Linear(prev_dim, hidden_dim))
            layers_value.append(activation_func())
            prev_dim = hidden_dim
        
        layers_policy.append(nn.Linear(prev_dim, action_dim))
        layers_policy.append(nn.Softmax(dim=-1))
        layers_value.append(nn.Linear(prev_dim, 1))
        
        self.fc_policy = nn.Sequential(*layers_policy)
        self.fc_value = nn.Sequential(*layers_value)
    
    def forward(self, latent_state):
        policy = self.fc_policy(latent_state)
        value = self.fc_value(latent_state)
        return policy, value

class NeuralNetworkManager:
    """Manages the training and deployment of MuZero’s three neural networks."""
    def __init__(self, input_dim, latent_dim, action_dim, hidden_layers, activation_func, lr=0.001):
        self.representation_network = RepresentationNetwork(input_dim, latent_dim, hidden_layers, activation_func)
        self.dynamics_network = DynamicsNetwork(latent_dim, action_dim, hidden_layers, activation_func)
        self.prediction_network = PredictionNetwork(latent_dim, action_dim, hidden_layers, activation_func)
        
        self.optimizer = optim.Adam(
            list(self.representation_network.parameters()) +
            list(self.dynamics_network.parameters()) +
            list(self.prediction_network.parameters()), lr=lr)
    
    def train_step(self, batch):
        """Runs one step of training using backpropagation through time (BPTT)."""
        states, actions, policies, values, rewards = batch
        
        # Forward pass
        latent_states = self.representation_network(states)
        pred_policies, pred_values = self.prediction_network(latent_states)
        
        loss_policy = F.cross_entropy(pred_policies, policies)
        loss_value = F.mse_loss(pred_values.squeeze(), values)
        loss_reward = 0
        
        # Simulate dynamics
        for i in range(actions.shape[1]):  # Rollout through time
            latent_states, pred_rewards = self.dynamics_network(latent_states, actions[:, i])
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
            latent_state = self.representation_network(state)
            policy, value = self.prediction_network(latent_state)
        return policy, value

# Example Usage
if __name__ == "__main__":
    manager = NeuralNetworkManager(input_dim=10, latent_dim=16, action_dim=4, hidden_layers=[128, 64], activation_func=nn.ReLU)
    dummy_state = torch.randn(1, 10)
    policy, value = manager.predict(dummy_state)
    print("Policy:", policy)
    print("Value:", value)