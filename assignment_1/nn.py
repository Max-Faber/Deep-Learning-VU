import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(o_i, O):
    return math.exp(o_i) / sum([math.exp(o) for o in O])


def init_parameters():
    parameters = dict()

    # parameters['w1'] = [[1., 1., 1.], [-1., -1., -1.]]
    # parameters['b1'] = [0., 0., 0.]
    # parameters['k1'] = [0., 0., 0.]
    # parameters['h1'] = [0., 0., 0.]
    #
    # parameters['w2'] = [[1., -1., -1.], [1., -1., -1.]]
    # parameters['b2'] = [0., 0.]
    # parameters['k2'] = [0., 0.]
    # parameters['h2'] = [0., 0.]

    parameters['w'] = [[1., 1., 1.], [-1., -1., -1.]]
    parameters['b'] = [0., 0., 0.]
    parameters['k'] = [0., 0., 0.]
    parameters['h'] = [0., 0., 0.]

    parameters['v'] = [[1., -1., -1.], [1., -1., -1.]]
    parameters['c'] = [0., 0.]
    parameters['o'] = [0., 0.]
    parameters['y'] = [0., 0.]
    return parameters


def forward_pass(x, parameters):
    for j in range(len(parameters['k'])):
        for i in range(len(x)):
            parameters['k'][j] += parameters['w'][i][j] * x[i]
        parameters['k'][j] += parameters['b'][j]
        parameters['h'][j] = sigmoid(parameters['k'][j])

    for i in range(len(parameters['c'])):
        for j in range(len(parameters['h'])):
            parameters['o'][i] += parameters['h'][j] * parameters['v'][i][j]
        parameters['o'][i] += parameters['c'][i]
    for i in range(len(parameters['y'])):
        parameters['y'][i] += softmax(parameters['o'][i], parameters['o'])
    pass


def backward_pass(parameters):
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function/46028029#46028029
    grads = dict()

    grads['dy_o'] = [[], []]
    range_o = range(len(parameters['o']))
    for i in range_o:
        for j in range_o:
            if i == j:
                dy_o = parameters['o'][i] * (1. - parameters['o'][j])
            else:
                dy_o = -parameters['o'][i] * parameters['o'][j]
            grads['dy_o'][i].append(dy_o)

    grads['do_v'] = []
    grads['do_h'] = []
    range_dy_o = range(len(grads['dy_o']))
    range_v = range(len(parameters['v']))
    range_h = range(len(parameters['h']))
    for i in range_dy_o:
        do_v = 0.
        for j in range_v:
            do_v += grads['dy_o'][i][j] * parameters['h'][i]
        grads['do_v'].append(do_v)
    for i in range_dy_o:
        d = [0 for _ in range_h]
        for j in range_v:
            for k in range_h:
                d[k] += grads['dy_o'][i][j] * parameters['v'][i][k]
        grads['do_h'].append(d)

    pass


def neural_network():
    parameters = init_parameters()
    x = [1., -1.]
    y = [1., 0.]
    forward_pass(x, parameters)
    backward_pass(parameters)


if __name__ == '__main__':
    neural_network()