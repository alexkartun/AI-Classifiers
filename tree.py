import random as rd
import math
from collections import defaultdict


def major_class(data, attributes, target):
    """ calculating major class from data by taking the most frequent """
    freq = defaultdict(int)
    index = attributes.index(target)

    for entry in data:
        freq[entry[index]] += 1

    frequent_class = max(freq.items(), key=lambda item: item[1])[0]

    return frequent_class


def entropy(data, attributes, target):
    """ calculating the overall data entropy """
    freq = defaultdict(int)
    index = attributes.index(target)

    for entry in data:
        freq[entry[index]] += 1

    data_entropy = 0.

    for freq in freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


def info_gain(data, attributes, attr, target):
    """ calculating info gain of the attribute """
    freq = defaultdict(int)
    index = attributes.index(attr)

    for entry in data:
        freq[entry[index]] += 1

    subset_entropy = 0.

    for key in freq.keys():
        prob = freq[key] / sum(freq.values())
        data_subset = [entry for entry in data if entry[index] == key]
        subset_entropy += prob * entropy(data_subset, attributes, target)

    return entropy(data, attributes, target) - subset_entropy


def attribute_choose(data, attributes, target):
    """ generating best attribute by taking the one with the highest info gain """
    max_gain = 0
    best = attributes[0]

    for attr in attributes:
        # skip target attribute
        if attr != target:
            gain = info_gain(data, attributes, attr, target)
            if gain > max_gain:
                max_gain = gain
                best = attr

    return best


def get_values(data, attributes, attr):
    """ generating all unique values of attribute by using set """
    index = attributes.index(attr)
    return set([entry[index] for entry in data])


def get_data(data, attributes, best_attr, val):
    """ generating new data set according to the best attribute and his value """
    new_data = []
    index = attributes.index(best_attr)

    for entry in data:
        if entry[index] == val:
            new_entry = []
            for i, value in enumerate(entry):
                # skip the column of best attribute
                if i != index:
                    new_entry.append(value)
            new_data.append(tuple(new_entry))
    return new_data


class DecisionTree(object):
    def __init__(self, decision=None, value=None, is_leaf=False):
        self.is_leaf = is_leaf
        self.decisions = {}
        self.decision = decision
        self.value = value


def print_tree(tree, depth=0):
    """ printing the tree recursively """
    if tree.is_leaf:
        return tree.value
    reprs = []
    items = tree.decisions.items()
    # sort the decisions in alphabetic manner by their decision value
    sorted_items = sorted(items, key=lambda item: item[0])
    for value, sub_tree in sorted_items:
        if sub_tree.is_leaf:
            sub_tree_repr = '{}{}{}={}:{}'.format('\t' * depth, '|' * (depth > 0), tree.decision, value,
                                                  print_tree(sub_tree, depth + 1))
        else:
            sub_tree_repr = '{}{}{}={}\n{}'.format('\t' * depth, '|' * (depth > 0), tree.decision, value,
                                                   print_tree(sub_tree, depth + 1))
        reprs.append(sub_tree_repr)
    return '\n'.join(reprs)


def build_tree(data, attributes, target):
    """ building the decision tree by id3 algorithm """
    values = [entry[attributes.index(target)] for entry in data]
    default = major_class(data, attributes, target)

    # data is empty or in attributes remained only the target attribute
    if not data or len(attributes) == 1:
        return DecisionTree(value=default, is_leaf=True)
    # all values are the same so return this value
    elif values.count(values[0]) == len(values):
        return DecisionTree(value=values[0], is_leaf=True)
    else:
        # choosing best attribute
        best_attr = attribute_choose(data, attributes, target)
        # generate new decision root
        tree = DecisionTree(decision=best_attr)
        # iterate over each value of best attribute
        for val in get_values(data, attributes, best_attr):
            # generate new data
            new_data = get_data(data, attributes, best_attr, val)
            # generate clone of attributes
            new_attributes = attributes.copy()
            # remove best attribute from attributes
            new_attributes.remove(best_attr)
            # call the function recursively with new data set and new attributes
            sub_tree = build_tree(new_data, new_attributes, target)
            # assign decision output of the value in tree's decisions map
            tree.decisions[val] = sub_tree

    return tree


def predict(example, attributes, tree):
    """ predicting the output of the tree on this specific example """
    # leaf return his value
    if tree.is_leaf:
        return tree.value
    # tree's decision
    decision = tree.decision
    # example's value on the decision attribute
    value = example[attributes.index(decision)]
    # calling the function recursively with the decisions branch according to the value
    return predict(example, attributes, tree.decisions[value])


def calculate_accuracy(predictions, gold_labels):
    """ calculating the accuracy of the predictions """
    correct = 0.

    for prediction, gold_label in zip(predictions, gold_labels):
        if prediction == gold_label:
            correct += 1

    print('accuracy is: {}'.format(correct / len(predictions)))
    return correct / len(predictions)


def main():
    training_set = []
    with open('.data/train.txt') as f:
        attributes = f.readline().split()
        for line in f.readlines():
            training_set.append(tuple(line.split()))
    target = attributes[-1]

    testing_set = []
    gold_labels = []
    with open('.data/test.txt') as f:
        for line in f.readlines()[1:]:
            values = line.split()
            testing_set.append(tuple(values[:-1]))
            gold_labels.append(values[-1])
    '''
    TRAINING
    '''
    rd.shuffle(training_set)
    tree = build_tree(training_set, attributes, target)
    print(print_tree(tree))

    '''
    TESTING
    '''
    predictions = []
    for example in testing_set:
        prediction = predict(example, attributes, tree)
        predictions.append(prediction)

    accuracy = calculate_accuracy(predictions, gold_labels)


if __name__ == "__main__":
    main()
