import yaml

import Game


class ANN:
    # This is a placeholder class for the Artificial Neural Network for the typehinting. 
    # Replace wiht actual ANN class when it is implemented.
    pass

def do_bptt():
    """
    This function should implement the backpropagation through time algorithm.
    """
    pass

def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('/C:/Users/hafst/MuZero/config.yaml')

class Reinforcement_Learning_System:
    def __init__(self):
        self.episode_history = []
        pass

    def episode_loop(self) -> tuple[ANN, ANN, ANN]: 
        """
        This should return the trained ANN objects for the Q-function, the policy and the value function.
        """

        # init ANN objects
        NNr = ANN()
        NNd = ANN()
        NNp = ANN()

        for episode in range(config.get('number_of_episodes')):
            game = Game()
            for k in range(config.get('number_of_steps_in_episode')):
                
                for m in range(config.get('number_of_MTC_simulations')):
                    pass

                
            pass

            self.episode_history.append(episode)
            if episode % 10 == 0:
                do_bptt()
        
        return NNr, NNd, NNp