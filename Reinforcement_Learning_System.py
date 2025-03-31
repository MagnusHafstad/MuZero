import yaml

from Game import Snake
from U_tree import U_tree
import numpy as np
from neural_network_manager import *
from typing import Callable
import matplotlib.pyplot as plt


def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file) 
    return config

config = load_config('./config.yaml')
nn_config = load_config('./nn_config.yaml')

class Reinforcement_Learning_System:
    def __init__(self, game:Callable):
        self.episode_history = []
        self.game = game
        pass

    def get_episode_history(self):
        return self.episode_history
    
    def init_new_game(self):
        game = self.game(config.get('game_size'), head=config.get('head')) ### Her er vi avhengig av hvilket spill det er. Ikke GENERELL KODE!
        return game
    
    def plot_metrics(self):
            
        state = []
        actions = []
        policies = []
        values = []
        rewards = []
        score = []
        survival= []

        for ep in self.episode_history:
            survival.append(len(ep))
            score.append(ep[-1][-1])
            # state.append(ep[0])
            # values.append(ep[1])
            # policies.append(ep[2])
            # actions.append(ep[3])
            # rewards.append(ep[4])
            # score.append(ep[5])

        X = np.linspace(1,len(self.episode_history), len(self.episode_history))
        plt.plot(X, survival, label ="survival") #kan kanskje være en ide å se på et gjennomsnitt
        plt.plot(X, np.array(score)-2, label ="fruit eaten")
        for i in range(config["train_config"]["batch_size"], len(X), config["train_config"]["batch_size"]):
            plt.axvline(x = i, ymin = 0.7, ymax = 1,  color = 'b')
        plt.xlabel("Episodes")
        plt.legend()
        plt.show()

    def plot_loss(self, NNr,NNd,NNp):
        X = np.linspace(1,len(NNr.loss), len(NNr.loss)) 
        #plt.plot(X, NNr.loss, label = "NNr")
        plt.plot(X, NNd.loss, label = "NNd")
        plt.plot(X, NNp.loss, label = "NNp")
        plt.legend()
        plt.xlabel("Training session")
        plt.ylabel("Loss")
        plt.show()

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
        

    def episode_loop(self) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]: 
        """|
        This should return the trained ANN objects for the Q-function, the policy and the value function.
        """

        # init ANN objects
        NNr = RepresentationNetwork()
        if nn_config["representation"]["load_path"]:
            NNr.load_state_dict(torch.load(nn_config["representation"]["load"] +'.pth'))
        NNd = DynamicsNetwork()
        if nn_config["dynamics"]["load_path"]:
            NNd.load_state_dict(torch.load(nn_config["dynamics"]["load"]+'.pth'))
        NNp = PredictionNetwork()
        if nn_config["prediction"]["load_path"]:
            NNp.load_state_dict(torch.load(nn_config["prediction"]["load"]+'.pth'))

        actions = config.get('set_of_actions')
        
        for episode_nr in range(config["train_config"]['number_of_episodes']):

            #Makes a new game for each episode
            game = self.init_new_game()  #Snake(config.get('game_size'), head=config.get('head')) <-- old code in case it don't work
            episode_data = []
            real_game_states = np.zeros((config["train_config"]['number_of_steps_in_episode']+1, config.get('game_size'), config.get('game_size')))
            real_game_states[0] = game.board.copy() #TODO: Fix this, it is probably wrong

            for k in range(config["train_config"]['number_of_steps_in_episode']):
                #makes a new abstract state for each step
                abstract_state = NNr.forward(real_game_states[k].flatten())
                u_tree = U_tree(abstract_state, config["train_config"]["max_depth"], actions)
                #Runs MCTS
                for m in range(config["train_config"]['number_of_MTC_simulations']):
                    u_tree.MCTS(NNd, NNp)
                #Gets the final policy for the step
                final_policy_for_step = u_tree.normalize_visits() #higly suspect
                #Saves episode data
                root_value = u_tree.get_root_value()
                next_action = u_tree.get_action(final_policy_for_step)
                next_state, next_reward = game.simulate_game_step(real_game_states[k], next_action)
                #next_state = next_state.detach().numpy()
                real_game_states[k+1]= next_state
                episode_data.append([real_game_states[k],root_value, final_policy_for_step, next_action, next_reward, game.get_score()])
                if game.status == "game_over":
                    break
            self.episode_history.append(episode_data)
            #does backpropagation
            if len(self.episode_history) % config['train_config']['training_interval'] == 0:
                do_bptt(NNr, NNd, NNp, self.episode_history, config['train_config']['batch_size']) ###########OBS!!!!!!!!!!!!!!
        
        return NNr, NNd, NNp
    
system = Reinforcement_Learning_System(Snake)


NNr, NNd, NNp = system.episode_loop()
system.plot_metrics()
system.plot_loss(NNr, NNd, NNp)




# Save the models
# torch.save(NNr.state_dict(), nn_config["representation"]["save_path"]+'.pth')
# if nn_config["representation"]["save_path"]:
#     NNr.load_state_dict(torch.save(NNr.state_dict(), nn_config["representation"]["save_path"] +'.pth'))

# if nn_config["dynamics"]["save_path"]:
#     NNd.load_state_dict(torch.save(NNd.state_dict(), nn_config["dynamics"]["save_path"]+'.pth'))

# if nn_config["prediction"]["save_path"]:
#     torch.save(NNp.state_dict(), nn_config["prediction"]["save_path"]+'.pth')
