from collections import defaultdict
import math


"""
ARGUMENTS
"""
train_path = 'train.txt'
test_path = 'test.txt'
output_tree_path = 'output_tree.txt'
output_path = 'output.txt'
k = 5


"""
UTILS
"""


def generate_training_data(training_file_path):
    """ generating train set, attributes list and target attribute """
    training_set = []
    with open(training_file_path) as f:
        attributes = f.readline().split()
        for line in f.readlines():
            training_set.append(tuple(line.split()))
    target = attributes[-1]

    return training_set, attributes, target


def generate_testing_data(test_file_path):
    """ generating testing set and gold labels """
    testing_set = []
    gold_labels = []
    with open(test_file_path) as f:
        for line in f.readlines()[1:]:
            values = line.split()
            testing_set.append(tuple(values[:-1]))
            gold_labels.append(values[-1])

    return testing_set, gold_labels


def extract_positive_target(data):
    """ generate positive value of this specific data set """

    if data[0][-1] == 'true' or data[0][-1] == 'false':
        positive_target = 'true'
    elif data[0][-1] == '1' or data[0][-1] == '0':
        positive_target = '1'
    else:
        positive_target = 'yes'
    return positive_target


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


def get_values(data, attributes, attr):
    """ generating all unique values of attribute by using set """
    index = attributes.index(attr)
    return set([entry[index] for entry in data])


def get_all_possible_values(data, attributes):
    """ generate of all possible values of each attribute """
    possible_values = defaultdict(set)

    for entry in data:
        for index, value in enumerate(entry[:-1]):
            possible_values[attributes[index]].add(value)

    return possible_values


def generate_attributes_vocabulary(data, attributes):
    vocab = defaultdict(set)

    for example in data:
        for attribute_index, value in enumerate(example[:-1]):
            vocab[attributes[attribute_index]].add(value)

    return vocab


def calculate_accuracy(predictions, gold_labels):
    """ calculating the accuracy of the predictions """
    correct = 0.

    for prediction, gold_label in zip(predictions, gold_labels):
        if prediction == gold_label:
            correct += 1

    accuracy = round(correct / len(predictions), 2)
    return accuracy


def output_tree(tree_representation, tree_output_path):
    """ save tree tree representation to output file """

    with open(tree_output_path, 'w') as f:
        f.write(tree_representation)


def output_predictions(tree_data, knn_data, nb_data, predictions_len, output_file_path):
    """ save all the models predictions to the output file """
    """ tree_data is in format of tuple: (tree_predictions, tree_accuracy) and etc. for each one of them """

    with open(output_file_path, 'w') as f:

        f.write('{}\t{}\t{}\t{}\n'.format('Num', 'DT', 'KNN', 'naiveBase'))
        for index in range(predictions_len):
            f.write('{}\t{}\t{}\t{}\n'.format(index + 1, tree_data[0][index], knn_data[0][index], nb_data[0][index]))

        f.write('{}\t{}\t{}\t{}'.format('', str(tree_data[1]), str(knn_data[1]), str(nb_data[1])))


"""
Decision Tree
"""


def major_class(data, attributes, target, positive_target):
    """ calculating major class from data by taking the most frequent """
    freq = defaultdict(int)
    index = attributes.index(target)

    for entry in data:
        freq[entry[index]] += 1

    # calculating max frequent class for breaking the tie will be taken positive target
    frequent_class = max(freq.items(), key=lambda item: (item[1], item[0] == positive_target))[0]

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


class DecisionTree(object):
    def __init__(self, decision=None, value=None, is_leaf=False):
        self.is_leaf = is_leaf
        self.decisions = {}
        self.decision = decision
        self.value = value


def get_tree_representation(tree, depth=0):
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
                                                  get_tree_representation(sub_tree, depth + 1))
        else:
            sub_tree_repr = '{}{}{}={}\n{}'.format('\t' * depth, '|' * (depth > 0), tree.decision, value,
                                                   get_tree_representation(sub_tree, depth + 1))
        reprs.append(sub_tree_repr)
    return '\n'.join(reprs)


def build_tree(data, attributes, target, possible_values, positive_target):
    """ building the decision tree by id3 algorithm """
    values = [entry[attributes.index(target)] for entry in data]
    default = major_class(data, attributes, target, positive_target)

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

        best_attr_values = get_values(data, attributes, best_attr)
        best_attr_possible_values = possible_values.get(best_attr)
        # iterate over each value of best attribute
        for val in best_attr_values:
            # generate new data
            new_data = get_data(data, attributes, best_attr, val)
            # generate clone of attributes
            new_attributes = list(attributes)
            # remove best attribute from attributes
            new_attributes.remove(best_attr)
            # call the function recursively with new data set and new attributes
            sub_tree = build_tree(new_data, new_attributes, target, possible_values, positive_target)
            # assign decision output of the value in tree's decisions map
            tree.decisions[val] = sub_tree

        # iterate over all other values of best attribute that don't appear in data set
        for val in best_attr_possible_values - best_attr_values:
            tree.decisions[val] = DecisionTree(value=default, is_leaf=True)
    return tree


def predict_on_tree(example, attributes, tree):
    """ predicting the output of the tree on this specific example """
    # leaf return his value
    if tree.is_leaf:
        return tree.value
    # tree's decision
    decision = tree.decision
    # example's value on the decision attribute
    value = example[attributes.index(decision)]
    # calling the function recursively with the decisions branch according to the value
    return predict_on_tree(example, attributes, tree.decisions[value])


"""
KNN
"""


def hamming_distance(test_example, training_example):
    """ return the Hamming distance between examples """

    return sum(test_attr != train_attr for test_attr, train_attr in zip(test_example, training_example))


def predict_knn(test_example, training_set, k):
    """ predicting the output by knn algorithm """
    distances = []
    # calculate distances for each example in train and save as tuple (index, target, hamming)
    for index, train_example in enumerate(training_set):
        distances.append((index, train_example[-1], hamming_distance(test_example, train_example[:-1])))
    # sort the list of tuples (key, distance) by second value in ascending order and break tie by their input index
    sorted_distances = sorted(distances, key=lambda item: (item[2], item[0]))

    # get top k rows from the sorted array
    targets = [target for i, target in enumerate(sorted_distances) if i < k]

    # calculate frequency of each class in targets
    freq = defaultdict(int)

    # generate frequencies of each of the targets
    for target in targets:
        freq[target[1]] += 1
    # get the most frequent class of these rows
    frequent_class = max(freq.items(), key=lambda item: item[1])[0]
    return frequent_class


"""
Naive Base
"""


def calculate_freq(data, attributes, target, value):
    """ calculate count of all values that equal to specific value, checking values only on target's index """
    freq = 0.
    index = attributes.index(target)

    for entry in data:
        if entry[index] == value:
            freq += 1

    return freq


def predict_nb(example, training_set, attributes, target):
    """ predicting the output by nb algorithm """
    attr_vocab = generate_attributes_vocabulary(training_set, attributes)
    values = get_values(training_set, attributes, target)
    prior_class_frequencies = {value: calculate_freq(training_set, attributes, target, value)
                               for value in values}
    probabilities = []
    for value in values:
        # prior probability
        prob = prior_class_frequencies[value] / len(training_set)
        for index, attribute in enumerate(example):
            # generate new data set
            new_data_set = [entry for entry in training_set if entry[index] == attribute]
            # calculate prob + smoothing
            prob *= (calculate_freq(new_data_set, attributes, target, value) + 1) / \
                    (prior_class_frequencies[value] + len(attr_vocab[attributes[index]]))
        probabilities.append((value, prob))

    max_prob_class = max(probabilities, key=lambda item: item[1])[0]
    return max_prob_class


"""
DATA
"""


training_set, attributes, target = generate_training_data(train_path)
testing_set, gold_labels = generate_testing_data(test_path)
possible_values = get_all_possible_values(training_set, attributes)
positive_target = extract_positive_target(training_set)

'''
TRAINING
'''

tree = build_tree(training_set, attributes, target, possible_values, positive_target)

'''
TESTING
'''


tree_predictions = []
knn_predictions = []
nb_predictions = []

# predicting
for example in testing_set:
    tree_predictions.append(predict_on_tree(example, attributes, tree))
    knn_predictions.append(predict_knn(example, training_set, k))
    nb_predictions.append(predict_nb(example, training_set, attributes, target))

# calculate accuracies
tree_accuracy = calculate_accuracy(tree_predictions, gold_labels)
knn_accuracy = calculate_accuracy(knn_predictions, gold_labels)
nb_accuracy = calculate_accuracy(nb_predictions, gold_labels)

# output tree to file
output_tree(get_tree_representation(tree), output_tree_path)

# output predictions to file
output_predictions((tree_predictions, tree_accuracy), (knn_predictions, knn_accuracy), (nb_predictions, nb_accuracy),
                   len(gold_labels), output_path)
