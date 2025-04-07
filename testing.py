
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import numpy as np
import random

from neural_network_manager import *

NNr = RepresentationNetwork()
state = [[0,0,0,0,0],
        [0,0,0,0,0],
        [1,2,0,0,-1],
        [0,0,0,0,0],
        [0,0,0,0,0]]
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).view(1, -1)  # Flatten the state to match input dimensions
abstract_state = NNr(state_tensor)

optimizer = optim.SGD(NNr.parameters(), lr=0.001)
random_target = torch.tensor([3, 7, 14, 2, 9, 18, 21, 6, 12, 25, 1, 4, 17, 10, 8, 5, 11, 20, 13, 16, 15, 22, 19, 24, 23], dtype=torch.float32).view(1, -1)  # Match target shape with output
loss_fn = nn.MSELoss()
loss = loss_fn(abstract_state, random_target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Gradient NNr: {NNr.parameters().__next__().grad}")



