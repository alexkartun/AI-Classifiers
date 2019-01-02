from collections import defaultdict


def calculate_freq(data, attributes, target, value):
    freq = 0.
    index = attributes.index(target)

    for entry in data:
        if entry[index] == value:
            freq += 1

    return freq


def generate_vocabulary(data):
    vocab = defaultdict(int)

    for example in data:
        for attribute in example[:-1]:
            vocab[attribute] += 1

    return vocab


def get_values(data, attributes, attr):
    """ generating all unique values of attribute by using set """
    index = attributes.index(attr)
    return set([entry[index] for entry in data])


def predict(example, training_set, vocab, attributes, target):
    """ predicting the output by nb algorithm """
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
                    (prior_class_frequencies[value] + vocab[attribute])
        probabilities.append((value, prob))

    max_prob_class = max(probabilities, key=lambda item: item[1])[0]
    return max_prob_class


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
    vocab = generate_vocabulary(training_set)

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
    for example in testing_set:
        prediction = predict(example, training_set, vocab, attributes, target)
        predictions.append(prediction)

    accuracy = calculate_accuracy(predictions, gold_labels)


if __name__ == "__main__":
    main()
