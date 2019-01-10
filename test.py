import decision_tree as dt
import knn
import nb
import utils as ut
import random as rd
import argparse

"""
ARGUMENTS
"""

parser = argparse.ArgumentParser(description='AI models')

# data paths
parser.add_argument("--train_path", type=str, default='.data/train.txt', help="Train data path")
parser.add_argument("--test_path", type=str, default='.data/test.txt', help="Test data path")
parser.add_argument("--output_tree", type=str, default='.data/output_tree.txt', help="Output tree path")
parser.add_argument("--output_path", type=str, default='.data/output.txt', help="Output path")

# testing
parser.add_argument("--k", type=int, default=5, help="Knn argument")

"""
CONFIG
"""

config = parser.parse_args()

training_set, attributes, target = ut.generate_training_data(config.train_path)
testing_set, gold_labels = ut.generate_testing_data(config.test_path)

'''
TRAINING
'''
tree = dt.build_tree(training_set, attributes, target)
# output tree to file
ut.output_tree(dt.get_tree_representation(tree), config.output_tree)

'''
TESTING
'''

tree_predictions = []
knn_predictions = []
nb_predictions = []

# predicting
for example in testing_set:
    tree_predictions.append(dt.predict(example, attributes, tree))
    knn_predictions.append(knn.predict(example, training_set, config.k))
    nb_predictions.append(nb.predict(example, training_set, attributes, target))

# calculate accuracies
tree_accuracy = ut.calculate_accuracy(tree_predictions, gold_labels)
knn_accuracy = ut.calculate_accuracy(knn_predictions, gold_labels)
nb_accuracy = ut.calculate_accuracy(nb_predictions, gold_labels)

# output tree to file
ut.output_tree(dt.get_tree_representation(tree), config.output_tree)

# output predictions to file
ut.output_predictions((tree_predictions, tree_accuracy), (knn_predictions, knn_accuracy), (nb_predictions, nb_accuracy),
                      len(gold_labels), config.output_path)
