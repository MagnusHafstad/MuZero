
class U_tree():
    def __init__(self, abstract_state):
        self.root = Tree_node(abstract_state, None, 0, 0)

    
    def search_to_leaf(self):
        pass


class Tree_node():
    def __init__(self, state, parent, reward, depth):
        self.state = state
        self.children = []
        self.parent = parent
        self.visit_count = 0
        self.reward = reward
        self.depth = depth

    def add_child(self, state, parent, reward):

        self.children.append(Tree_node(self, state, parent, reward, self.depth + 1))
        
