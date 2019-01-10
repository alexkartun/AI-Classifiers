from collections import defaultdict


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
