import numpy as np
from data import load_mnist

def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1. / (1. + np.exp(-x))

def softmax(o):
    """

    :param o:
    :return:
    """
    return np.exp(o) / np.sum(np.exp(o))

def init_parameters_example():
    """

    :return:
    """
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
    params['k'] = np.matmul(X, params['w']) + params['b']
    params['h'] = sigmoid(params['k'])
    params['o'] = np.matmul(params['h'], params['v']) + params['c']
    params['y'] = softmax(params['o'])
    return params

def backward_pass(X, Y, params, cache, layer_sizes):
    """

    :param X:
    :param Y:
    :param params:
    :param cache:
    :param layer_sizes:
    :return:
    """
    # Convert the target class to a one-hot encoding
    Y_one_hot = np.zeros(layer_sizes['n_outputs'])
    Y_one_hot[Y] = 1
    # Backward pass of second layer
    dy_do = params['y'] - Y_one_hot
    dy_dc = np.copy(dy_do)
    do_dv = np.outer(params['h'], dy_do)
    do_dh = np.matmul(params['v'], dy_do)
    # Backward pass of first layer
    dh_dk = do_dh * params['h'] * (1. - params['h'])
    dk_dw = np.outer(X, dh_dk)
    dk_db = np.copy(dh_dk)
    grads = {
        'dy_dv': do_dv,
        'dy_dc': dy_dc,
        'dk_dw': dk_dw,
        'dk_db': dk_db
    }
    return grads

def propagate(params, X_batch, Y_batch, layer_sizes):
    """
    Performs a forward- & backward pass and compute the cost
    :param params: Dictionary of parameters which are necessary for the forward- & backward pass
    :param X_batch: Batch of lists of input values
    :param Y_batch: Batch of integer values indicating the target classes
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :return: The cost (integer) and gradients (dictionary) which will be used to update the weights
    """
    grads_sum = {
        'dy_dv': np.zeros((layer_sizes['n_hidden'], layer_sizes['n_outputs'])),
        'dy_dc': np.zeros(layer_sizes['n_outputs']),
        'dk_dw': np.zeros((layer_sizes['n_inputs'], layer_sizes['n_hidden'])),
        'dk_db': np.zeros(layer_sizes['n_hidden'])
    }
    costs = []
    batch_size = len(X_batch)
    for X, Y in zip(X_batch, Y_batch):
        # Perform a forward pass given x and the current parameters
        cache = forward_pass(X=X, params=params)
        # Calculate the cost using the predicted probability of the target class
        prob_y = cache['y'][Y]
        cost = -np.log(prob_y)
        costs.append(cost)
        # Perform a backward pass given x, y and the current parameters
        grads = backward_pass(X=X, Y=Y, params=params, cache=cache, layer_sizes=layer_sizes)
        # grads = backward_pass(X=X, Y=Y, params=params, cache=cache, layer_sizes=layer_sizes)
        grads_sum = { key: value + grads_sum[key] for key, value in grads.items() }
    avg_grads = { key: value / batch_size for key, value in grads_sum.items() }
    return costs, avg_grads

def update_weights(params, grads, learning_rate):
    params['v'] -= learning_rate * grads['dy_dv']
    params['c'] -= learning_rate * grads['dy_dc']
    params['w'] -= learning_rate * grads['dk_dw']
    params['b'] -= learning_rate * grads['dk_db']
    return params

def normalize(X):
    """
    Normalize a list of values between 0 and 1
    :param X: List of values to normalize
    :return: Normalized list of values
    """
    max_val = X.max()
    return X / max_val

def optimize(params, X, Y, n_epochs, learning_rate, layer_sizes, batch_size):
    """
    Trains the model using the given training data and target values for a given no. epochs and using a given learning rate
    :param params: Parameters (weights) of the model
    :param X: List of instances containing the features to train on
    :param Y: List of target values of the instances
    :param n_epochs: No. epochs to train the model
    :param learning_rate: Learning rate for updating the weights
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :param batch_size: Size of the batches being fed into the network before updating the weights
    :return: Dictionary mapping the name of the layer with its corresponding weights after training the model
    """
    cost_per_epoch = []
    n_samples = len(X)
    for epoch in range(1, n_epochs + 1):
        cost_per_instance = np.array([])
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:min(i + batch_size, n_samples)]
            Y_batch = Y[i:min(i + batch_size, n_samples)]
            costs, grads = propagate(params=params, X_batch=X_batch, Y_batch=Y_batch, layer_sizes=layer_sizes)
            cost_per_instance = np.append(cost_per_instance, costs)
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
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), num_cls = load_mnist()
    # Normalize the training data
    xtrain_norm = normalize(xtrain)
    del xtrain
    # Define the layer sizes
    layer_sizes = {
        'n_inputs': xtrain_norm.shape[1],
        'n_hidden': 300,
        'n_outputs': ytrain.max() + 1
    }
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1., 1.))
    return optimize(params=params, X=xtrain_norm, Y=ytrain, n_epochs=1000, learning_rate=0.01, layer_sizes=layer_sizes, batch_size=128)

if __name__ == '__main__':
    neural_network()
