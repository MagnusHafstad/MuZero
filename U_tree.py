import numpy as np

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
    def __init__(self, abstract_state):
        self.root = Tree_node(abstract_state, None, 0, 0)

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


