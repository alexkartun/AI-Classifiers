from collections import defaultdict


def hamming_distance(test_example, training_example):
    """ return the Hamming distance between examples """

    return sum(test_attr != train_attr for test_attr, train_attr in zip(test_example, training_example))


def predict(test_example, training_set, k):
    """ predicting the output by knn algorithm """
    distances = []
    # calculate distances for each example in train
    for train_example in training_set:
        distances.append((train_example[-1], hamming_distance(test_example, train_example[:-1])))
    # sort the list of tuples (key, distance) by second value in ascending order
    sorted_distances = sorted(distances, key=lambda item: item[1])
    # get top k rows from the sorted array
    targets = [target for i, target in enumerate(sorted_distances) if i < k]

    # calculate frequency of each class in targets
    freq = defaultdict(int)

    for target in targets:
        freq[target] += 1
    # get the most frequent class of these rows
    frequent_class = max(freq.items(), key=lambda item: item[1])[0][0]
    return frequent_class
