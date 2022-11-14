import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(o):
    return np.exp(o) / np.sum(np.exp(o))


def softmax_grad(y):
    s = y.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def init_parameters_example():
    return {
        'w': np.array([[1., 1., 1.], [-1., -1., -1.]]),
        'b': np.array([0., 0., 0.]),
        'k': np.array([0., 0., 0.]),
        'h': np.array([0., 0., 0.]),
        'v': np.array([[1., 1.], [-1., -1.], [-1., -1.]]),
        'c': np.array([0., 0.]),
        'o': np.array([0., 0.]),
        'y': np.array([0., 0.])
    }

def init_parameters():
    shape_w = (2, 3)
    shape_v = (3, 2)
    min_weight_range = 0.
    max_weight_range = 1.
    # random.seed(63)

    return {
        'w': np.array([[random.uniform(min_weight_range, max_weight_range) for _ in range(shape_w[1])] for _ in range(shape_w[0])]),
        'b': np.array([0., 0., 0.]),
        'k': np.array([0., 0., 0.]),
        'h': np.array([0., 0., 0.]),
        'v': np.array([[random.uniform(min_weight_range, max_weight_range) for _ in range(shape_v[1])] for _ in range(shape_v[0])]),
        'c': np.array([0., 0.]),
        'o': np.array([0., 0.]),
        'y': np.array([0., 0.])
    }


def forward_pass(x, parameters):
    parameters['k'] = x.dot(parameters['w']) + parameters['b']
    parameters['h'] = sigmoid(parameters['k'])
    parameters['o'] = parameters['v'].T.dot(parameters['h']) + parameters['c']
    parameters['y'] = softmax(parameters['o'])
    return parameters

def print_shapes(grads):
    for key, value in grads.items():
        print(f"Shape of '{key}': {value.shape}")

def update_weights(params, grads, learning_rate):
    params['v'] -= learning_rate * grads['o_v']
    params['c'] -= learning_rate * grads['y_c']
    params['w'] -= learning_rate * grads['k_w']
    params['b'] -= learning_rate * grads['h_b']
    return params

def backward_pass(x, y, parameters):
    grads = {}
    Y = np.zeros(len(parameters['y']))
    Y[y] = 1.

    grads['y_o'] = parameters['y'] - Y
    grads['y_c'] = np.copy(grads['y_o'])
    grads['o_v'] = np.outer(parameters['h'], grads['y_o'])
    grads['o_h'] = np.multiply(parameters['v'], grads['y_o'])

    grads['h_k'] = np.sum(np.outer(sigmoid(parameters['h']) * (1. - sigmoid(parameters['h'])), grads['o_h']), axis=1)
    grads['k_w'] = np.outer(x, grads['h_k'])
    grads['h_b'] = np.copy(grads['h_k'])
    return grads


def neural_network():
    x = np.array([1., -1.])
    parameters = init_parameters_example()

    forward_pass(x, parameters)
    backward_pass(x, 0, parameters)


if __name__ == '__main__':
    neural_network()
