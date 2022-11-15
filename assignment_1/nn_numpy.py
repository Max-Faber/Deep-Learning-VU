import numpy as np
from assignment_1.data import load_mnist

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
        'v': np.array([[1., 1.], [-1., -1.], [-1., -1.]]),
        'c': np.array([0., 0.])
    }

def init_parameters(layer_sizes, weight_range):
    """
    Initializes the parameters of the neural network using the given layer sizes and weight range
    :param layer_sizes: Dictionary mapping the size (no. neurons) per layer
    :param weight_range: Tuple indicating the min. and max. range of the weights which are initialized randomly
    :return: Dictionary mapping the layer names to its corresponding parameters (weights)
    """
    return {
        'w': np.random.uniform(weight_range[0], weight_range[1], (layer_sizes['n_inputs'], layer_sizes['n_hidden'])),
        'b': np.zeros(layer_sizes['n_hidden']),
        'v': np.random.uniform(weight_range[0], weight_range[1], (layer_sizes['n_hidden'], layer_sizes['n_outputs'])),
        'c': np.zeros(layer_sizes['n_outputs'])
    }

def forward_pass(X, params):
    params['k'] = X.dot(params['w']) + params['b']
    params['h'] = sigmoid(params['k'])
    params['o'] = params['v'].T.dot(params['h']) + params['c']
    params['y'] = softmax(params['o'])
    return params

def backward_pass(X, Y, params):
    Y_one_hot = np.zeros(len(params['y']))
    Y_one_hot[Y] = 1

    dl_dy = np.zeros(len(params['y']))
    dl_dy[Y] = -1. / params['y'][Y]

    dy_do = params['y'] - Y_one_hot
    dy_do = (dl_dy * params['y'] * (1. - params['y'])) + np.flip(dl_dy * -params['y'] * params['y'])
    dy_dc = np.copy(dy_do)
    do_dv = np.outer(params['h'], dy_do)
    do_dh = np.multiply(params['v'], dy_do)

    dh_dk = np.sum(np.outer(params['h'] * (1. - params['h']), do_dh), axis=1)
    dk_dw = np.outer(X, dh_dk)
    dk_db = np.copy(dh_dk)
    grads = {
        'dy_dv': do_dv,
        'dy_dc': dy_dc,
        'dk_dw': dk_dw,
        'dk_db': dk_db
    }
    return grads

def propagate(params, x, y, layer_sizes):
    """
    Performs a forward- & backward pass and compute the cost
    :param params: Dictionary of parameters which are necessary for the forward- & backward pass
    :param x: Array of input values
    :param y: Integer value indicating the target class
    :param layer_sizes: Dictionary mapping the size for each layer
    :return: The cost (integer) and gradients (dictionary) which will be used to update the weights
    """
    # Perform a forward pass given x and the current parameters
    cache = forward_pass(X=x, params=params, layer_sizes=layer_sizes)
    # Calculate the cost using the predicted probability of the target class
    prob_y = cache['y'][y]
    cost = -math.log(prob_y)
    # Perform a backward pass given x, y and the current parameters
    grads = backward_pass(x=x, y=y, params=params, cache=cache)
    return cost, grads

def update_weights(params, grads, learning_rate):
    params['v'] -= learning_rate * grads['o_v']
    params['c'] -= learning_rate * grads['y_c']
    params['w'] -= learning_rate * grads['k_w']
    params['b'] -= learning_rate * grads['h_b']
    return params

def normalize(X):
    """
    Normalize a list of values between 0 and 1
    :param X: List of values to normalize
    :return: Normalized list of values
    """
    max_val = X.max()
    return X / max_val

def optimize(params, X, Y, n_epochs, learning_rate, layer_sizes):
    """
    Trains the model using the given training data and target values for a given no. epochs and using a given learning rate
    :param params: Parameters (weights) of the model
    :param X: List of instances containing the features to train on
    :param Y: List of target values of the instances
    :param n_epochs: No. epochs to train the model
    :param learning_rate: Learning rate for updating the weights
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :return: Dictionary mapping the name of the layer with its corresponding weights after training the model
    """
    cost_per_epoch = []
    for epoch in range(n_epochs):
        cost_per_instance = []
        for x, y in zip(X, Y):
            cost, grads = propagate(params=params, x=x, y=y, layer_sizes=layer_sizes)
            cost_per_instance.append(cost)
            params = update_weights(params=params, grads=grads, learning_rate=learning_rate)
        mean_cost = sum(cost_per_instance) / len(cost_per_instance)
        cost_per_epoch.append(mean_cost)
        print(f'Epoch {epoch}, cost: {mean_cost:.5f}')
    return params

def neural_network():
    """
    Constructs and trains a neural network
    :return: The parameters (weights) of the neural network
    """
    X = np.array([1., -1.])
    params = init_parameters_example()
    forward_pass(X=X, params=params)
    backward_pass(X=X, Y=0, params=params)

    # Define the layer sizes
    layer_sizes = {
        'n_inputs': 2,
        'n_hidden': 300,
        'n_outputs': 10
    }
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1., 1.))
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), num_cls = load_mnist()
    # Normalize the training data
    xtrain_norm, ytrain_norm = normalize(xtrain), normalize(ytrain)

    return params

if __name__ == '__main__':
    neural_network()
