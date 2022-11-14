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

    for j in range_k:
        parameters['k'][j] = 0.
        parameters['h'][j] = 0.
        for i in range_x:
            parameters['k'][j] += parameters['w'][i][j] * x[i]
        parameters['k'][j] += parameters['b'][i]
        parameters['h'][j] = sigmoid(parameters['k'][i])

    for i in range_o:
        parameters['o'][i] = 0.
        for j in range_h:
            parameters['o'][i] += parameters['h'][j] * parameters['v'][j][i]
        parameters['o'][i] += parameters['c'][i]

    for i in range_y:
        parameters['y'][i] = softmax(parameters['o'][i], parameters['o'])

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

def backward_pass(x, y, params):
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function/46028029#46028029
    ranges = get_ranges(params)
    dl_dy = [0. for _ in ranges['y']]
    dy_do = [0. for _ in ranges['y']]
    do_dv = [[0. for _ in ranges['o']] for _ in ranges['v']]
    do_dh = [0. for _ in ranges['h']]
    dh_dk = [0. for _ in ranges['h']]
    dk_dw = [[0. for _ in ranges['k']] for _ in ranges['w']]

    dl_dy[y] = -1 / params['y'][y]
    for j in ranges['y']:
        for i in ranges['o']:
            if i == j:
                dy_do[i] += dl_dy[j] * params['y'][j] * (1. - params['y'][i])
            else:
                dy_do[i] += dl_dy[j] * -params['y'][j] * params['y'][i]
    dy_dc = [d for d in dy_do]

    for j in ranges['o']:
        for i in ranges['v']:
            do_dv[i][j] += dy_do[j] * params['h'][i]
            do_dh[i] += dy_do[j] * params['v'][i][j]

    for i in ranges['h']:
        dh_dk[i] += do_dh[i] * sigmoid(params['h'][i]) * (1. - sigmoid(params['h'][i]))
    dk_db = [d for d in dh_dk]

    for j in ranges['k']:
        for i in ranges['w']:
            dk_dw[i][j] += dh_dk[j] * x[i]

    grads = {
        'dy_dv': do_dv,
        'dy_dc': dy_dc,
        'dk_dw': dk_dw,
        'dk_db': dk_db
    }
    return grads

def neural_network():
    x = [1., -1.]
    parameters = init_parameters()
    parameters = forward_pass(x, parameters)
    grads = backward_pass(x, 0, parameters)
    print(grads)

if __name__ == '__main__':
    neural_network()
