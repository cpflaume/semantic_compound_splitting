from annoy import AnnoyIndex

input("<Enter> to create tree.")
tree = AnnoyIndex(500)

input("<Enter> to load tree.")
tree.load("test_tree.ann")

input("<Enter> to load 10,000 vectors.")
q = True
while q:

    for i in range(10000):
        tree.get_item_vector(i)

    resp = input("<Enter> to load 10,000 vectors.")
    if resp.strip() == "q":
        q = False

input("<Enter> to unload tree.")

tree.unload("test_tree.ann")

input("done.")
tree.load("test_tree.ann")

input("<Enter> to load 10,000 vectors.")
q = True
while q:

    for i in range(10000):
        tree.get_item_vector(i)

    resp = input("<Enter> to load 10,000 vectors.")
    if resp.strip() == "q":
        q = False

input("<Enter> to delete tree.")

del tree

input("done.")
