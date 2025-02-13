import torch
import torch.nn as nn

class RepresentationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RepresentationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(DynamicsNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size + action_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        return x

class PredictionNetwork(nn.Module):
    def __init__(self, hidden_size, action_size, value_size):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size, action_size)
        self.fc2 = nn.Linear(hidden_size, value_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        policy = self.fc1(x)

        policy = self.softmax(policy)
        value = self.fc2(x)
        return policy, value

# Example usage
input_size = 10
hidden_size = 64
action_size = 4
value_size = 1

representation_net = RepresentationNetwork(input_size, hidden_size)
dynamics_net = DynamicsNetwork(hidden_size, action_size)
prediction_net = PredictionNetwork(hidden_size, action_size, value_size)

# Generate random input
input_data = torch.randn(1, input_size)

# Forward pass through the networks
hidden_state = representation_net(input_data)
action = torch.randn(1, action_size)
next_hidden_state = dynamics_net(hidden_state, action)
policy, value = prediction_net(next_hidden_state)

print("Policy:", policy)
print("Value:", value)