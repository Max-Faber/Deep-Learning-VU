import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(o_i, O):
    return math.exp(o_i) / sum([math.exp(o) for o in O])

def init_parameters():
    return {
        'w': [[1., 1., 1.], [-1., -1., -1.]],
        'b': [0., 0., 0.],
        'k': [0., 0., 0.],
        'h': [0., 0., 0.],
        'v': [[1., 1.], [-1., -1.], [-1., -1.]],
        'c': [0., 0.],
        'o': [0., 0.],
        'y': [0., 0.]
    }

def forward_pass(x, parameters):
    range_x = range(len(x))
    range_k = range(len(parameters['k']))
    range_h = range(len(parameters['h']))
    range_o = range(len(parameters['o']))
    range_y = range(len(parameters['y']))

    for i in range_k:
        for j in range_x:
            parameters['k'][i] += parameters['w'][j][i] * x[j]
        parameters['k'][i] += parameters['b'][i]
        parameters['h'][i] = sigmoid(parameters['k'][i])

    for i in range_o:
        for j in range_h:
            parameters['o'][i] += parameters['h'][j] * parameters['v'][j][i]
        parameters['o'][i] += parameters['c'][i]

    for i in range_y:
        parameters['y'][i] += softmax(parameters['o'][i], parameters['o'])
    return parameters


# Get rid of numpy before handing this in
import numpy as np

def print_shapes(grads):
    for key, value in grads.items():
        print(f"Shape of '{key}': {np.array(value).shape}")

def get_ranges(params):
    ranges = dict()
    for layer_name, values_dict in params.items():
        ranges[layer_name] = range(len(values_dict))
    return ranges

def backward_pass(x, params):
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function/46028029#46028029
    params['x'] = x
    ranges = get_ranges(params)
    grads = {
        'y_c': [0. for _ in ranges['y']],
        'y_o': [0. for _ in ranges['y']],
        'o_v': [[0. for _ in ranges['o']] for _ in ranges['v']],
        'o_h': [[0. for _ in ranges['o']] for _ in ranges['h']],
        'h_b': [0. for _ in ranges['h']],
        'h_k': [0. for _ in ranges['h']],
        'k_w': [[0. for _ in ranges['k']] for _ in ranges['w']]
    }

    for i in ranges['y']:
        for j in ranges['y']:
            if i != j:
                grads['y_o'][0] += -params['y'][i] * params['y'][j]
            else:
                grads['y_o'][1] += params['y'][i] * (1. - params['y'][j])
    grads['y_c'] = [d for d in grads['y_o']]

    for i in ranges['v']:
        for j in ranges['o']:
            grads['o_v'][i][j] += grads['y_o'][j] * params['h'][i]
            grads['o_h'][i][j] += grads['y_o'][j] * params['v'][i][j]

    for i in ranges['h']:
        for j in ranges['o']:
            grads['h_k'][i] += grads['o_h'][i][j] * sigmoid(params['h'][i]) * (1. - sigmoid(params['h'][i]))
    grads['h_b'] = [d for d in grads['h_k']]

    for i in ranges['w']:
        for j in ranges['k']:
            grads['k_w'][i][j] += grads['h_k'][j] * x[i]
    return grads

def neural_network():
    x = [1., -1.]
    parameters = init_parameters()
    parameters = forward_pass(x, parameters)
    grads = backward_pass(x, parameters)
    print(grads)

if __name__ == '__main__':
    neural_network()
