import numpy as np
import random
import torch
from typing import Callable
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

class Tree_node():
    def __init__(self, state, parent, reward, depth, status):
        self.state = state
        self.children = []
        self.parent = parent
        self.visit_count = 0
        self.reward = reward
        self.depth = depth
        self.status = status

    def add_child(self, state, parent, reward, status):
        child = Tree_node(state, parent, reward, self.depth + 1, status)
        self.children.append(child)
        

class U_tree():
    def __init__(self, abstract_state, d_max, actions):
        self.root = Tree_node(abstract_state.clone(), None, 0, 0, "playing")
        self.d_max = d_max
        self.actions = actions
        # self.game = game


    def get_root_value(self):
        return self.root.reward
    
    def tree_policy(self, node:Tree_node):
        """
        Use upper confidence bounds(UCB) as tree policy
        """
        c = config["exploration_rate"]
        visit_count = node.visit_count
        if node.visit_count == 0:
            visit_count = 0.00001
        return node.reward + c * np.sqrt(np.log(node.parent.visit_count)/visit_count)

    
    def search_to_leaf(self) -> Tree_node:
        if len(self.root.children) == 0:
            return self.root 

        current_node = self.root
        visited_nodes = set()

        while len(current_node.children) != 0:
            if current_node in visited_nodes:  # Prevent infinite loops
                break
            visited_nodes.add(current_node)
            current_node = self.select_child_node(current_node)

        #print(f"Current Node Depth: {current_node.depth}, Children: {len(current_node.children)}")     

        return current_node
    

    def select_child_node(self, node)-> Tree_node:
        current_UCB = float('-inf')
        for child in node.children:
            ucb = self.tree_policy(child)
            if ucb > current_UCB:
                current_UCB = ucb
                best_child = child

        if best_child is None:
            raise ValueError(f"Last parent {node.depth} has children {node.children}")

        return best_child

### only for debug
    def print_tree(self, node, level=0):
        print(" " * (level * 4) + f"depth: {node.depth} Reward: {node.reward}, Visits: {node.visit_count}") #State: {node.state},
        for child in node.children:
            self.print_tree(child, level + 1)

    def write_tree_to_file(self, node, file, level=0):
        file.write(" " * (level * 4) + f"depth: {node.depth} Reward: {node.reward}, Visits: {node.visit_count}, State:  {node.state}, status: {node.status}  \n")
        for child in node.children:
            self.write_tree_to_file(child, file, level + 1)

    def save_tree(self, filename="tree.txt"):
        with open(filename, "w") as file:
            self.write_tree_to_file(self.root, file)
  

    def MCTS(self, calc_next_state: Callable, get_policy:Callable):
        """
        Do one full MC run through the tree
        """
        leaf_node = self.search_to_leaf()
        if leaf_node.status == "game_over":
            leaf_node.visit_count = 1
            leaf_node.reward = config["game_over_reward"]
            return
        
        if len(leaf_node.children)!=0:
            raise ValueError("Leaf node should not have children")
        
        for i,action in enumerate(self.actions):
            new_state, reward, status  = calc_next_state(leaf_node.state, [action])
            #new_state = new_state.detach().numpy() # OBS i tilfelle vi ikke får gradienter, sjekk denne!
            #reward = int(reward.item())
            leaf_node.add_child(new_state,leaf_node,reward, status)

        child = random.choice(leaf_node.children)
        accum_reward = self.do_rollout(child, self.d_max - child.depth, get_policy, calc_next_state)
        self.do_backpropagation(child, sum(accum_reward)) 

    
    def do_rollout(self, node:Tree_node, depth, get_policy:Callable, calc_next_state:Callable) -> list[float]:
        """
        do further simulation untill desired depth of tree
        NOT DONE!!!
        """
        accum_reward = []
        if config["use_NN"]:
            state = node.state.clone()
        else:
            state = node.state.copy()
        status = "playing"
        if not config["use_NN"]:
            status = node.status
        for _ in range(depth):
            if status != "playing":
                break
            if config["use_NN"]:
                state_policy, state_value = get_policy(state)
                state_policy = state_policy.detach().numpy()[0]
            else:
                state_policy, state_value = get_policy(node)

            action = self.get_action(state_policy) 

            state, reward, status = calc_next_state(state, [action])
            
            #state = state.detach().numpy() # OBS i tilfelle vi ikke får gradienter, sjekk denne!
            #reward = int(reward.item())
            accum_reward.append(reward)
        if config["use_NN"]:
            state_policy, state_value = get_policy(state)
            state_policy = state_policy.detach().numpy()[0]
        else:
            state_policy, state_value = get_policy(node)

        #state_value = state_value.item()
        accum_reward.append(state_value)
        return accum_reward

    def do_backpropagation(self, node, reward: float, discount_rate=1):
        """
        updates the reward value of the nodes
        """
    
        while node != None:
            node.visit_count += 1
            node.reward += reward 
            reward = node.reward * discount_rate  
            node = node.parent  


    def get_action(self, policy): 
        """
        Policy should be a normalized 1-d vector
        """
        #policy = policy.detach().numpy()[0]
        action = np.random.choice(len(policy), p=policy)
        return action
    
    def get_final_action(self, policy):
        action = np.argmax(policy)
        return action
    
    def normalize_visits(self):
        """
        This function should normalize the visits of the children of a node.
        """
        policy = []
        for child in self.root.children:
            policy.append(child.visit_count)
        policy = np.array(policy)
        policy = policy/sum(policy)
        print(policy)
        return policy




