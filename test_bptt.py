
from neural_network_manager import *
import yaml
from Reinforcement_Learning_System import Reinforcement_Learning_System


with open('nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)
    

system = Reinforcement_Learning_System()
system.episode_loop()

NNr = RepresentationNetwork()
if nn_config["representation"]["load"]:
    NNr.load_state_dict(torch.load('NNr.pth'))
NNd = DynamicsNetwork()
if nn_config["dynamics"]["load"]:
    NNd.load_state_dict(torch.load('NNd.pth'))
NNp = PredictionNetwork()
if nn_config["prediction"]["load"]:
    NNp.load_state_dict(torch.load('NNp.pth'))
print("bppt")
do_bptt(NNr, NNd, NNp, system.episode_history, 10)