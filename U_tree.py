

class Tree_node():
    def __init__(self, state, parent, reward, depth):
        self.state = state
        self.children = []
        self.parent = parent
        self.visit_count = 0
        self.reward = reward
        self.depth = depth

    def add_child(self, state, parent, reward):

        self.children.append(Tree_node(state, parent, reward, self.depth + 1))
        

class U_tree():
    def __init__(self, abstract_state):
        self.root = Tree_node(abstract_state, None, 0, 0)

    
    def search_to_leaf(self) -> Tree_node:
        if len(self.root.children) == 0:
            return self.root 
        current_node = self.root

        while len(current_node.children) != 0:
            current_node = current_node.children[0]
        
        return current_node
            
            



def depth_first_search(node, target):
    if node is None:
        return False
    if node.depth == target:
        return True
    for child in node.children:
        if depth_first_search(child, target):
            return True
    return False


tree = U_tree(4)
print("hei")
for i in range(3):
    print("ho")
    tree.root.add_child(2,tree.root,2)

tempchild = tree.root.children

for node in tempchild:
    node.add_child(3,node,3)

    print(node)
print("ferdig init")

print(tree.search_to_leaf().state)



