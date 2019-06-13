import numpy as np
import neural_network as nn
import utils


def main():
    # loading the datasets.
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")

    train_x, train_y = utils.shuffle(train_x, train_y)

    # data normalization
    utils.normalize_data(train_x, 'min_max')

    # Create Neaural network
    net = nn.NeuralNetwork(train_x.shape)

    # Training the model
    net.train(train_x, train_y)

    # Testing
    result = net.test(test_x)
    # Extract test result to file.
    file = open('test_y', 'w+')
    for y in result:
        file.write("{}\n".format(y))


if __name__ == '__main__':
    main()
