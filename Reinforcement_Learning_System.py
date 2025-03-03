import yaml

from Game import Snake
from U_tree import U_tree
import numpy as np
from neural_network_manager import *


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

config = load_config('./config.yaml')

class Reinforcement_Learning_System:
    def __init__(self):
        self.episode_history = []
        pass

    def generate_real_game_states(self, episode_nr) -> list[tuple]:
        """
        This function should generate real game states by playing the game.
        """
        game_states = []    
        for episode in range(len(self.episode_history)):
            game = Snake()
            # TODO: Run through the game and save the game states
            game_states.append((game.board, game.snake))
        return game_states
    
    def normalize_visits():
        """
        This function should normalize the visits of the children of a node.
        """
        pass
    
    def select_state(final_policy_for_step):
        """
        This function should select the next state based on the policy.
        """
        #TODO: Test/Implement this function. This is pure GPT
        action = np.random.choice(len(final_policy_for_step), p=final_policy_for_step)
        return action
        

    def episode_loop(self) -> tuple[ANN, ANN, ANN]: 
        """|
        This should return the trained ANN objects for the Q-function, the policy and the value function.
        """

        # init ANN objects
        NNr = RepresentationNetwork()
        NNd = DynamicsNetwork()
        NNp = PredictionNetwork()
        actions = config.get('set_of_actions')
        for episode_nr in range(config["train_config"]['number_of_episodes']):
            game = Snake(config.get('game_size'), head=config.get('head'))
            episode_data = []
            real_game_states = np.zeros((config["train_config"]['number_of_steps_in_episode']+1, config.get('game_size'), config.get('game_size')))
            real_game_states[0] = game.get_real_nn_game_state() #TODO: Fix this, it is probably wrong
            for k in range(config["train_config"]['number_of_steps_in_episode']):
                
                abstract_state = NNr.forward(real_game_states[k])
                u_tree = U_tree(abstract_state, config["train_config"]["max_depth"], actions)
                for m in range(config["train_config"]['number_of_MTC_simulations']):
                    u_tree.MCTS(actions, NNd, NNr, NNp)
                final_policy_for_step = u_tree.normalize_visits() #higly suspect
                root_value = u_tree.get_root_value()
                next_action = u_tree.get_action(final_policy_for_step)
                next_state, next_reward = game.simulate_game_step(real_game_states[k], next_action)
                real_game_states[k+1]= next_state
                episode_data.append([real_game_states[-1],root_value, final_policy_for_step, next_action, next_reward])
            self.episode_history.append(episode_data)
            # if len(episode_data) % config(['train_config']['training_interval']) == 0:
            #     do_bptt() 
        
        return NNr, NNd, NNp
    
system = Reinforcement_Learning_System()
system.episode_loop()
