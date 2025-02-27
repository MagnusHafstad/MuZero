import numpy as np
import random
import torch

class Tree_node():
    def __init__(self, state, parent, reward, depth):
        self.state = state
        self.children = []
        self.parent = parent
        self.visit_count = 1
        self.reward = reward
        self.depth = depth

    def add_child(self, state, parent, reward):

        self.children.append(Tree_node(state, parent, reward, self.depth + 1))
        

class U_tree():
    def __init__(self, abstract_state, d_max, actions):
        self.root = Tree_node(abstract_state, None, 0, 0)
        self.d_max = d_max
        self.actions = actions

    def tree_policy(self, node:Tree_node):
        """
        Use upper confidence bounds(UCB) as tree policy
        """
        c = 1
        return node.reward + c* np.sqrt(np.log(node.parent.visit_count)/node.visit_count)

    
    def search_to_leaf(self) -> Tree_node:
        if len(self.root.children) == 0:
            return self.root 
        current_node = self.root

        while len(current_node.children) != 0:
            current_node = self.select_child_node(current_node)     
        return current_node
    
    def select_child_node(self, node)-> Tree_node:
        current_UCB = -100
        for child in node.children:
            UCB = self.tree_policy(child)
            if UCB > current_UCB:
                current_UCB = UCB
                node = child
        return node

### only for debug
    def print_tree(self, node, level=0):
        print(" " * (level * 4) + f"State: {node.state}, Reward: {node.reward}, Visits: {node.visit_count}")
        for child in node.children:
            self.print_tree(child, level + 1)      

    def MCTS(self, actions, NNd, NNs, NNp):
        """
        Do one full MC run through the tree
        """
        leaf_node = self.search_to_leaf()
        
        for i,action in enumerate(self.actions):
            new_state, reward = NNd(leaf_node.state, [action])
            new_state = new_state.detach().numpy() # OBS i tilfelle vi ikke får gradienter, sjekk denne!
            reward = int(reward.item())
            
            leaf_node.add_child(new_state,leaf_node,reward)

        child = random.choice(leaf_node.children)
        accum_reward = self.do_rollout(child, self.d_max - child.depth, NNp, NNd, NNs)
        self.do_backpropagation(child, self.root, accum_reward) 

    
    def do_rollout(self, node:Tree_node, depth, NNp, NNd, NNs) -> list[float]:
        """
        do further simulation untill desired depth of tree
        NOT DONE!!!
        """
        accum_reward = []
        state = node.state
        for _ in range(depth):
            state_policy, state_value = NNp(torch.tensor(state))
            state_policy = state_policy.detach().numpy()
            action = self.get_action(state_policy) 
            state, reward = NNd(state, [action])

            state = state.detach().numpy() # OBS i tilfelle vi ikke får gradienter, sjekk denne!
            reward = int(reward.item())
            accum_reward.append(reward)

        state_policy, state_value = NNp(torch.tensor(state))

        state_value = state_value.item()
        accum_reward.append(state_value)
        return accum_reward

    def do_backpropagation(self, node, goal_node, accum_rewards: list):
        """
        updates the reward value of the nodes
        """
        node.visit_count += 1
        node.reward = sum(accum_rewards)
        accum_rewards.append(node.reward)
        if node != goal_node:
            self.do_backpropagation(node.parent,goal_node, accum_rewards) 


        pass

    def get_action(self, policy): 
        """
        Policy should be a normalized 1-d vector
        """
        action = np.random.choice(self.actions, p=policy)
        return action




