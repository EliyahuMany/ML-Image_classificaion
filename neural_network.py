import numpy as np
import utils

VALIDATION_SIZE = 0.2
HIDDEN_LAYER_SIZE = 100
EPOCHS = 100
ETA = 0.005
MIN_RANDOM = -0.08
MAX_RANDOM = 0.08
CLASSES = 10


class NeuralNetwork:
    def __init__(self, shape):
        """
        The neural network const.
        :param shape: the shape of the training dataset.
        """
        self.weights1 = np.random.uniform(MIN_RANDOM, MAX_RANDOM, [HIDDEN_LAYER_SIZE, shape[1]])
        self.weights2 = np.random.uniform(MIN_RANDOM, MAX_RANDOM, [CLASSES, HIDDEN_LAYER_SIZE])
        self.bias1 = np.random.uniform(MIN_RANDOM, MAX_RANDOM, [HIDDEN_LAYER_SIZE, 1])
        self.bias2 = np.random.uniform(MIN_RANDOM, MAX_RANDOM, [CLASSES, 1])

    def feedforward(self, x):
        """
        Feed forward function calculating the nn nodes from the input
        to the hidden layer to the output layer.
        :param x: train data set
        :return: z1, first hidden layer values before use the activation function.
                 h1, the hidden layer values after using the activation function.
                 z2, the output layer values.
        """
        x = np.transpose(x)
        z1 = np.dot(self.weights1, x) + self.bias1
        h1 = utils.sigmoid(z1)
        z2 = np.dot(self.weights2, h1) + self.bias2
        return z1, h1, z2

    def backprop(self, x, y, z1, h1, z2, probs):
        """
        Back propagation function calculate the derivatives of each layer
        for specific data set value.
        :param x: one of the dataset values.
        :param y: the correlated class.
        :param z1: first hidden layer values before use the activation function.
        :param h1: the hidden layer values after using the activation function.
        :param z2: the output layer values.
        :param probs: list of probabilities for each class.
        """
        y_vec = np.zeros((CLASSES, 1))
        y_vec[int(y)] = 1

        dz2 = (probs - y_vec)
        dw2 = np.dot(dz2, h1.T)
        db2 = dz2
        dz1 = np.dot(self.weights2.T, (probs - y_vec)) * utils.sigmoid(z1) * (1 - utils.sigmoid(z1))
        dw1 = np.dot(dz1, x)
        db1 = dz1

        self.update_weights(dw1, dw2, db1, db2)

    def train(self, x_set, y_set):
        """
        Train function, training the model by splitting first the
        train dataset to train and validation, for each epoch we use
        shuffle for the original dataset and split it again. at the end
        of each epoch we use validation function to check accuracy and
        average loss for the specific epoch.
        :param x_set: the complete training dataset.
        :param y_set: the correlated classes.
        """
        loss_sum = 0
        for i in range(EPOCHS):
            x_set, y_set = utils.shuffle(x_set, y_set)
            train_x, train_y, val_x, val_y = utils.split_validation(x_set, y_set, VALIDATION_SIZE)
            train_x, train_y = utils.shuffle(train_x, train_y)

            # running of each example from the train dataset.
            for x, y in zip(train_x, train_y):
                x = np.reshape(x, (1, x.shape[0]))
                z1, h1, z2 = self.feedforward(x)
                probs = utils.softmax(self.weights2, h1, self.bias2, CLASSES)
                loss = utils.loss(probs[int(y)])
                loss_sum += loss
                self.backprop(x, y, z1, h1, z2, probs)
            val_loss, acc = self.validation(val_x, val_y)
            # print("Epoch Num {}\nLoss = {}\tAccuracy = {}".format(i, val_loss, acc))

    def update_weights(self, dw1, dw2, db1, db2):
        """
        Update weights function receive the derivatives and updating
        each one of the weights and bias.
        :param dw1: input->hidden layer weights.
        :param dw2: hidden layer->output layer weights.
        :param db1: input->hidden layer bias weight.
        :param db2: hidden layer->output layer bias weight.
        """
        self.weights1 = self.weights1 - ETA * dw1
        self.weights2 = self.weights2 - ETA * dw2
        self.bias1 = self.bias1 - ETA * db1
        self.bias2 = self.bias2 - ETA * db2

    def validation(self, val_x, val_y):
        """
        Validation function using for internal testing of the
        model training.
        :param val_x: validation data set.
        :param val_y: validation correlated classes.
        :return: avg_loss, the average loss of the validation set
                 acc, the accuracy of the validation set (success/validation set length).
        """
        loss_sum = 0
        success = 0

        for x, y in zip(val_x, val_y):
            x = np.reshape(x, (1, x.shape[0]))
            z1, h1, z2 = self.feedforward(x)
            probs = utils.softmax(self.weights2, h1, self.bias2, CLASSES)
            loss_sum += utils.loss(probs[int(y)])
            y_hat = utils.softmax(self.weights2, h1, self.bias2, CLASSES).argmax(axis=0)

            # the model guess is correct.
            if y == y_hat[0]:
                success += 1
        avg_loss = loss_sum / len(val_y)
        acc = success / len(val_y)
        return avg_loss, acc

    def test(self, test_set):
        """
        Test function using to predict test dataset.
        :param test_set: the testing dataset.
        :return: the model predictions.
        """
        results = []

        for x in test_set:
            x = np.reshape(x, (1, x.shape[0]))
            z1, h1, z2 = self.feedforward(x)
            results.append(utils.softmax(self.weights2, h1, self.bias2, CLASSES).argmax(axis=0)[0])

        return results
