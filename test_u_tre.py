from U_tree import U_tree

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
