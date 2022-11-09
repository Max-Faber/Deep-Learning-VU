import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(o_i, O):
    return math.exp(o_i) / sum([math.exp(o) for o in O])


def init_parameters():
    parameters = dict()

    # parameters['w'] = [[1., 1., 1.], [-1., -1., -1.]]
    # parameters['b'] = [0., 0., 0.]
    # parameters['k'] = [0., 0., 0.]
    # parameters['h'] = [0., 0., 0.]
    #
    # parameters['v'] = [[1., -1., -1.], [1., -1., -1.]]
    # parameters['c'] = [0., 0.]
    # parameters['j'] = [0., 0.]
    # parameters['y'] = [0., 0.]

    parameters['w1'] = [[1., 1., 1.], [-1., -1., -1.]]
    parameters['b1'] = [0., 0., 0.]
    parameters['k1'] = [0., 0., 0.]
    parameters['h1'] = [0., 0., 0.]

    parameters['w2'] = [[1., -1., -1.], [1., -1., -1.]]
    parameters['b2'] = [0., 0.]
    parameters['k2'] = [0., 0.]
    parameters['h2'] = [0., 0.]
    return parameters


def forward_pass(x, parameters):
    for j in range(len(parameters['k1'])):
        for i in range(len(x)):
            parameters['k1'][j] += parameters['w1'][i][j] * x[i]
        parameters['k1'][j] += parameters['b1'][j]
        parameters['h1'][j] = sigmoid(parameters['k1'][j])

    for i in range(len(parameters['b2'])):
        for j in range(len(parameters['h1'])):
            parameters['k2'][i] += parameters['h1'][j] * parameters['w2'][i][j]
        parameters['k2'][i] += parameters['b2'][i]
    for i in range(len(parameters['h2'])):
        parameters['h2'][i] += softmax(parameters['k2'][i], parameters['k2'])
    pass


def backward_pass(parameters):
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function/46028029#46028029
    grads = dict()
    grads['dh2'] = [[], []]
    grads['dk2'] = []
    grads['dw2'] = []

    for i in range(len(parameters['h2'])):
        for j in range(len(parameters['h2'])):
            if i == j:
                dh2 = parameters['h2'][i] * (1 - parameters['h2'][j])
            else:
                dh2 = -parameters['h2'][i] * parameters['h2'][j]
            grads['dh2'][i].append(dh2)
    for i in range(len(parameters['k2'])):
        dk2 = 0.
        # dw2 = 0.
        for j in range(len(grads['dh2'][i])):
            dk2 += grads['dh2'][i][j] * parameters['k2'][i]
            # dw2 += grads['dh2'][i][j] * parameters['w2'][i]
        grads['dk2'].append(dk2)
        # grads['dw2'].append(dw2)
    pass


def neural_network():
    parameters = init_parameters()
    x = [1., -1.]
    y = [1., 0.]
    forward_pass(x, parameters)
    backward_pass(parameters)


if __name__ == '__main__':
    neural_network()