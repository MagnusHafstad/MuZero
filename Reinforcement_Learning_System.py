import yaml

import Game
import U_tree
import numpy as np

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
            game = Game()
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
        NNr = ANN()
        NNd = ANN()
        NNp = ANN()

        for episode_nr in range(config.get('number_of_episodes')):
            game = Game()
            episode_data = []
            real_game_states = game.get_game_state()
            for k in range(config.get('number_of_steps_in_episode')):
                
                abstract_state = NNr.forward(real_game_states)
                u_tree = U_tree(abstract_state)
                for m in range(config.get('number_of_MTC_simulations')):
                    leaf_node = u_tree.search_to_leaf()
                    for action in (config.get('set_of_actions')):
                        next_abstract_state, next_reward = NNd.forward(leaf_node, action)
                        leaf_node.add_child(next_abstract_state, next_reward)
                    
                    random_child = leaf_node.get_random_child()
                    accumulated_reward = do_rollout(random_child, config.get(['train_config']['max_depth']) - u_tree.get_depth(random_child), NNd, NNp)
                    do_backpropagation(random_child, abstract_state, accumulated_reward)
                final_policy_for_step = normalize_visits(leaf_node.get_visits()) #higly suspect
                root_value = u_tree.get_root_value()
                next_action = self.select_state(final_policy_for_step)
                next_state, next_reward = simulate_game_step(real_game_states[-1], next_action)
                real_game_states[k]= next_state
                episode_data.append([real_game_states[-1],root_value, final_policy_for_step, next_action, next_reward])
            self.episode_history.append(episode_data)
            if len(episode_data) % config.get(['train_config']['training_interval']) == 0:
                do_bptt()
        
        return NNr, NNd, NNp
    
    