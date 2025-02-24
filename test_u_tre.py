from U_tree import U_tree

def testTree():
    #Generate tree:
    tree = U_tree(1)
    for i in range(5):
        tree.root.add_child(i,tree.root,i)

    for child in tree.root.children:
        for i in range(3):
            child.add_child(i,child,i)

    tree.print_tree(tree.root, 1)

    #test search
    node = tree.search_to_leaf()
    print(f"State: {node.state}, Reward: {node.reward}, Visits: {node.visit_count}")
    print(node.parent.state)


testTree()