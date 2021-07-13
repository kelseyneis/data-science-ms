import numpy as np


class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """

    def __init__(
        self,
    ):
        self.feature = None  # index of the selected feature (for non-leaf node)
        self.class_label = None  # class label (for leaf node)
        self.left_child = None  # left child node
        self.right_child = None  # right child node


class Decision_tree:
    """
    Decision tree with binary features
    """

    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self, train_x, train_y):
        # construct the decision-tree with recursion
        self.root = generate_tree(train_x, train_y, self.min_entropy)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype("int")
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            cur_node = self.root
            while cur_node.class_label == None:
                if test_x[i][cur_node.feature] == 0:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            prediction[i] = cur_node.class_label
        return prediction


def split_indexes(data, feature, feature_value):
    return [i for i, val in enumerate(data[:, feature]) if val == feature_value]


def max_label(label):
    uniq = np.unique(label, return_counts=True)
    return uniq[0][np.argmax(uniq[1])]


def generate_tree(data, label, min_entropy):
    # initialize the current tree node
    cur_node = Tree_node()

    # compute the node entropy
    node_entropy = compute_node_entropy(label)

    # determine if the current node is a leaf node
    if node_entropy < min_entropy:
        # determine the class label for leaf node
        cur_node.class_label = max_label(label)
        return cur_node

    # select the feature that will best split the current non-leaf node
    selected_feature = select_feature(data, label)
    cur_node.feature = selected_feature
    # split the data based on the selected feature and start the next level of recursion
    left_child_indexes = split_indexes(data, cur_node.feature, 0)
    left_child_data = data[left_child_indexes]
    left_child_label = label[left_child_indexes]

    right_child_indexes = split_indexes(data, cur_node.feature, 1)
    right_child_data = data[right_child_indexes]
    right_child_label = label[right_child_indexes]

    cur_node.left_child = generate_tree(left_child_data, left_child_label, min_entropy)
    cur_node.right_child = generate_tree(right_child_data, right_child_label, min_entropy)

    return cur_node


def select_feature(data, label):
    # iterate through all features and compute their corresponding entropy
    best_feat = 0
    prev_entropy = 100

    for i in range(len(data[0])):
        # compute the entropy of splitting based on the selected features
        cur_entropy = compute_split_entropy(
            label[split_indexes(data, i, 0)], label[split_indexes(data, i, 1)]
        )
        if cur_entropy < prev_entropy:
            best_feat = i
            prev_entropy = cur_entropy
        # select the feature with minimum entropy

    return best_feat


def compute_split_entropy(left_y, right_y):
    # compute the entropy of a potential split, left_y and right_y are labels for the two splits
    split_entropy = 0  # placeholder
    split_entropy += (
        len(left_y) / (len(left_y) + len(right_y)) * compute_node_entropy(left_y)
    )
    split_entropy += (
        len(right_y) / (len(left_y) + len(right_y)) * compute_node_entropy(right_y)
    )

    return split_entropy


def compute_node_entropy(label):
    # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
    node_entropy = 0
    for i in np.unique(label):
        proportion = (label == i).sum() / len(label)
        node_entropy -= proportion * np.log2(proportion + 1e-15)

    return node_entropy
