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
    def __init__(self, abstract_state, d_max):
        self.root = Tree_node(abstract_state, None, 0, 0)
        self.d_max = d_max

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
        
        for i,action in enumerate(actions):
            new_state, reward = NNd(leaf_node.state, [action])
            new_state = new_state.detach().numpy() # OBS i tilfelle vi ikke får gradienter, sjekk denne!
            reward = int(reward.item())
            
            leaf_node.add_child(new_state,leaf_node,reward)
        child = random.choice(leaf_node.children)
        self.print_tree(child)
        #accum_reward = self.do_rollout(child, self.d_max - child.depth, NNd, NNs)  #ikke klar !!
        
        #self.do_backpropagation(child, self.root, accum_reward) # ikke klar!

    
    def do_rollout(self, NNp):
        accum_reward = []
        return accum_reward

    def do_backpropagation(self):
        pass



