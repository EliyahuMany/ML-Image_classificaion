import numpy as np


def normalize_data(x, normalize_method):
    """
    The function normalize the given dataset by one of the
    two methods (min max or mean normalization). (0-1)
    :param x: dataset to normalize.
    :param normalize_method: min max / mean normalization.
    :return: normalized dataset.
    """
    for i in range(0, x.shape[0]):
        min = x[i].min()
        max = x[i].max()

        if normalize_method == 'min_max':
            x[i] = (x[i] - min) / (max - min)
        elif normalize_method == 'mean':
            sum = 0
            for val in x[i]:
                sum += val
            avg = sum / len(x[i])
            x[i] = (x[i] - avg) / (max - min)
    return x


def normalize_probs(x):
    """
    The function normalize the predictions probabilities. (0-1)
    :param x: the predictions probabilities.
    :return: normalized predictions.
    """
    min = x.min()
    max = x.max()

    for i in range(len(x)):
        x[i] = (x[i] - min) / (max - min)

    return x


def sigmoid(param):
    """
    Sigmoid function, using as the model activation function.
    :param param: the input value to the hidden layer node.
    :return: the value after using sigmoid function.
    """
    return np.divide(1, (1 + np.exp(-param)))


def shuffle(x, y):
    """
    Shuffle function mixing the training dataset correlated to
    the classification dataset.
    :param x: training dataset.
    :param y: training classification dataset.
    :return: the mixed datasets.
    """
    shape = np.arange(x.shape[0])
    np.random.shuffle(shape)
    x = x[shape]
    y = y[shape]
    return x, y


def loss(predict):
    """
    Loos function, in our model we used Negative Log Likelihood (NLL).
    :param predict: the right class predict percents divided by 100. (0-1)
    :return: -log(predict).
    """
    return -np.log(predict)


def softmax(w, xt, b, size):
    """
    Softmax function responsible to normalize the probabilities such that
    sum of all prediction probabilities together equal to 1.
    :param w: weights of the hidden layer to the output layer.
    :param xt: input value to the hidden layer.
    :param b: the bias weight for the output layer.
    :param size: number of classes (outputs).
    :return: the predictions probabilities values after softmax.
    """
    sum = 0
    for j in range(size):
        sum += np.exp(np.dot(w[j], xt) + b[j])

    softmax_vec = np.zeros((size, 1))
    for i in range(size):
        softmax_vec[i] = (np.exp(np.dot(w[i], xt) + b[i])) / sum

    return softmax_vec


def split_validation(x, y, size):
    """
    the function splits the given dataset to training and validation datasets
    by the ratio size.
    :param x: training dataset.
    :param y: classification dataset.
    :param size: ratio of the desirable validation set.
    :return: train_x, train_y, training and classification dataset
             val_x, val_y, validation dataset.
    """
    val_size = int(x.shape[0] * size)
    val_x = x[-val_size:, :]
    val_y = y[-val_size:]
    train_x = x[:-val_size, :]
    train_y = y[:-val_size]
    return train_x, train_y, val_x, val_y
