import utils as ut


def calculate_freq(data, attributes, target, value):
    """ calculate count of all values that equal to specific value, checking values only on target's index """
    freq = 0.
    index = attributes.index(target)

    for entry in data:
        if entry[index] == value:
            freq += 1

    return freq


def predict(example, training_set, attributes, target):
    """ predicting the output by nb algorithm """
    attr_vocab = ut.generate_attributes_vocabulary(training_set, attributes)
    values = ut.get_values(training_set, attributes, target)
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
                    (prior_class_frequencies[value] + len(attr_vocab[attribute]))
        probabilities.append((value, prob))

    max_prob_class = max(probabilities, key=lambda item: item[1])[0]
    return max_prob_class
