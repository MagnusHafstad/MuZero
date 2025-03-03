from neural_network_manager import *
import yaml
hidden_layers = [{"dim": 10, "activation": "relu"}, {"dim": 7, "activation": "tanh"}]
with open('nn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config["representation"])
def test_representation():
    representation = RepresentationNetwork()
    print(representation)
    print(representation.forward(torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=torch.float32)))

def test_dynamics():
    dynamics = DynamicsNetwork()
    print(dynamics)
    print(dynamics.forward(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14], dtype=torch.float32), torch.tensor([15,16], dtype=torch.float32)))

def test_prediction():
    prediction = PredictionNetwork()
    print(prediction)
    print(prediction.forward(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14], dtype=torch.float32)))
    
test_representation()
test_dynamics()
test_prediction()