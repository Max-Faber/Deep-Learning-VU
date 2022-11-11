import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(o):
    return np.exp(o) / np.sum(np.exp(o))


def softmax_grad(y):
    s = y.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def init_parameters():
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


def forward_pass(x, parameters):
    parameters['k'] = x.dot(parameters['w']) + parameters['b']
    parameters['h'] = sigmoid(parameters['k'])
    parameters['o'] = parameters['v'].T.dot(parameters['h']) + parameters['c']
    parameters['y'] = softmax(parameters['o'])
    pass

def print_shapes(grads):
    for key, value in grads.items():
        print(f"Shape of '{key}': {value.shape}")

def backward_pass(x, parameters):
    grads = {}

    grads['y_o'] = np.array([-.5, .5])# softmax_grad(parameters['y'])
    # grads['y_o'] = softmax_grad(parameters['y'])
    grads['y_c'] = np.copy(grads['y_o'])
    grads['o_v'] = np.outer(parameters['h'], grads['y_o'])
    grads['o_h'] = np.multiply(parameters['v'], grads['y_o'])

    grads['h_k'] = np.multiply(sigmoid(parameters['h']) * (1. - sigmoid(parameters['h'])), grads['o_h'].T)
    grads['k_w'] = np.multiply(x, grads['h_k'].T)# np.outer(x, grads['h_k'])
    pass


def neural_network():
    x = np.array([1., -1.])
    parameters = init_parameters()

    forward_pass(x, parameters)
    backward_pass(x, parameters)


if __name__ == '__main__':
    neural_network()
