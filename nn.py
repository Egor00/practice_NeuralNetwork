import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def f(x):
    # approximated function
    return np.log(1 + np.e ** x)


class NeuralNetwork:
    def __init__(self, input_num, hidden_num, output_num, lern_rate):
        # The number of neurons of the input, hidden and output layers, respectively
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        # weight matrices
        self.weights1 = 2 * np.random.random((input_num + 1, hidden_num)) - 1
        self.weights2 = 2 * np.random.random((hidden_num, output_num)) - 1
        self.lern_rate = lern_rate

    def __call__(self, input_val):
        input_val = np.append(input_val, 1)
        layer1 = sigmoid(np.dot(input_val, self.weights1))
        output = sigmoid(np.dot(layer1, self.weights2))
        return output

    def load_weights(self):
        try:
            self.weights1 = np.load('w1.npy')
            self.weights2 = np.load('w2.npy')
            return True
        except FileNotFoundError:
            return False

    def save_weights(self):
        np.save('w1', self.weights1)
        np.save('w2', self.weights2)

    def train(self, input_, output_, iter_num=100):
        errors = []
        for i in range(iter_num):
            err = 0
            for x, y in zip(input_, output_):
                x = np.append(x, 1)
                hidden_layer = sigmoid(np.dot(x, self.weights1))
                output = sigmoid(np.dot(hidden_layer, self.weights2))

                # gradient descent method
                out_delta = self.lern_rate * (y - output) * sigmoid_derivative(output)
                l1_delta = self.lern_rate * out_delta * self.weights2.T * sigmoid_derivative(hidden_layer)

                self.weights2 += (hidden_layer * out_delta).reshape(self.weights2.shape)
                self.weights1 += np.dot(x.reshape((self.input_num + 1, 1)), l1_delta.reshape((1, self.hidden_num)))

                e = (y - output) ** 2
                err += e
            err /= iter_num
            errors.append(err)
        return errors


def plot(x, y):
    plt.plot(x, y)
    plt.grid(True)
    plt.show()


def test_show(nn_, err=0.333):
    points_num = 1000
    x = np.linspace(-10, 10, points_num)
    y_f = np.array([f(t) for t in x])

    x /= 10
    y_nn = np.array([float(nn_(t)) for t in x])
    y_nn *= 10
    x *= 10

    e = sum(abs(y_f - y_nn) < err)

    plt.plot(x, y_nn, 'g')
    plt.plot(x, y_f, 'b')
    plt.grid(True)

    plt.show()
    return e / points_num * 100


if __name__ == '__main__':
    # example of training and using
    np.seterr(all='ignore')
    np.random.seed(9)
    input_set = 20 * np.random.random_sample((400,)) - 10
    output_set = np.array([f(x) for x in input_set])
    input_set /= 10
    output_set /= 10

    nn = NeuralNetwork(1, 9, 1, 0.09)
    # nn.load_weights()
    nn.train(input_set, output_set, 100)
    # nn.save_weights()
    print(test_show(nn))
