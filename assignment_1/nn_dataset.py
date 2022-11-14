import random
import math
from data import load_synth
from nn import forward_pass, backward_pass, get_ranges
# from nn_numpy import forward_pass, backward_pass, update_weights

def init_parameters():
    shape_w = (2, 3)
    shape_v = (3, 2)
    min_weight_range = 0.
    max_weight_range = 1.
    # random.seed(63)

    return {
        # 'w': [[random.uniform(min_weight_range, max_weight_range) for _ in range(shape_w[1])] for _ in range(shape_w[0])],
        'w': [[-0.19, -0.65, 1.03], [1.74, -0.78, -0.88]],
        'b': [0., 0., 0.],
        'k': [0., 0., 0.],
        'h': [0., 0., 0.],
        # 'v': [[np.random.normal(min_weight_range, max_weight_range) for _ in range(shape_v[1])] for _ in range(shape_v[0])],
        'v': [[-0.01, 1.09], [-1.25, 0.04], [0.03, -0.25]],
        'c': [0., 0.],
        'o': [0., 0.],
        'y': [0., 0.]
    }

def propagate(params, x, y):
    params = forward_pass(x, params)
    predicted_y = params['y'][y]
    cost = -math.log(predicted_y)
    grads = backward_pass(x, y, params)
    return params, cost, grads

def update_weights(params, grads, learning_rate):
    for i in range(len(params['v'])):
        for j in range(len(params['v'][i])):
            params['v'][i][j] -= learning_rate * grads['dy_dv'][i][j]
    for i in range(len(params['c'])):
        params['c'][i] -= learning_rate * grads['dy_dc'][i]
    for i in range(len(params['w'])):
        for j in range(len(params['w'][i])):
            params['w'][i][j] -= learning_rate * grads['dk_dw'][i][j]
    for i in range(len(params['b'])):
        params['b'][i] -= learning_rate * grads['dk_db'][i]
    return params

def optimize(params, X, Y, n_epochs, learning_rate=0.01):
    cost_per_epoch = []
    for epoch in range(n_epochs):
        cost_per_instance = []
        for x, y in zip(X, Y):
            params, cost, grads = propagate(params=params, x=x, y=y)
            cost_per_instance.append(cost)
            params = update_weights(params=params, grads=grads, learning_rate=learning_rate)
        mean_cost = sum(cost_per_instance) / len(cost_per_instance)
        cost_per_epoch.append(mean_cost)
        print(f'Epoch {epoch}, cost: {mean_cost:.5f}')

import numpy as np

def convert_np(parameters):
    for key, val in parameters.items():
        parameters[key] = np.array(val)
    return parameters

if __name__ == '__main__':
    params = convert_np(init_parameters())
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
    optimize(params=params, X=xtrain, Y=ytrain, n_epochs=10000)
