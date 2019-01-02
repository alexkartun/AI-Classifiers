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

    '''
    CALCULATE FREQUENCY OF EACH CLASS IN TARGETS
    '''
    freq = defaultdict(int)

    for target in targets:
        freq[target] += 1
    # get the most frequent class of these rows
    frequent_class = max(freq.items(), key=lambda item: item[1])[0][0]
    return frequent_class


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
        for line in f.readlines()[1:]:
            training_set.append(tuple(line.split()))

    testing_set = []
    gold_labels = []
    with open('.data/test.txt') as f:
        for line in f.readlines()[1:]:
            values = line.split()
            testing_set.append(tuple(values[:-1]))
            gold_labels.append(values[-1])

    '''
    TESTING
    '''
    predictions = []
    k = 5
    for example in testing_set:
        prediction = predict(example, training_set, k)
        predictions.append(prediction)

    accuracy = calculate_accuracy(predictions, gold_labels)


if __name__ == "__main__":
    main()
