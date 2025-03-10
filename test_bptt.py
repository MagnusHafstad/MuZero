import unittest
from neural_network_manager import *
import yaml
with open('nn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)