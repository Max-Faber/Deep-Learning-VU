import random
import math
from data import load_synth
from nn import forward_pass, backward_pass

def init_parameters(layer_sizes, weight_range):
    """
    Initializes the parameters of the neural network using the given layer sizes and weight range
    :param layer_sizes: Dictionary mapping the size (no. neurons) per layer
    :param weight_range: Tuple indicating the min. and max. range of the weights which are initialized randomly
    :return:
    """
    return {
        'w': [[random.uniform(weight_range[0], weight_range[1]) for _ in range(layer_sizes['n_hidden'])] for _ in range(layer_sizes['n_inputs'])],
        'b': [0. for _ in range(layer_sizes['n_hidden'])],
        'v': [[random.uniform(weight_range[0], weight_range[1]) for _ in range(layer_sizes['n_outputs'])] for _ in range(layer_sizes['n_hidden'])],
        'c': [0. for _ in range(layer_sizes['n_outputs'])]
    }

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
    cache = forward_pass(x=x, params=params, layer_sizes=layer_sizes)
    # Calculate the cost using the predicted probability of the target class
    prob_y = cache['y'][y]
    cost = -math.log(prob_y)
    # Perform a backward pass given x, y and the current parameters
    grads = backward_pass(x=x, y=y, params=params, cache=cache, layer_sizes=layer_sizes)
    return cost, grads

def update_weights(params, grads, learning_rate, layer_sizes):
    """
    Updates the weights of the neural network
    :param params:
    :param grads:
    :param learning_rate:
    :param layer_sizes:
    :return:
    """
    for j in range(layer_sizes['n_hidden']):
        for i in range(layer_sizes['n_outputs']):
            params['v'][j][i] -= learning_rate * grads['dy_dv'][j][i]
    for i in range(layer_sizes['n_outputs']):
        params['c'][i] -= learning_rate * grads['dy_dc'][i]
    for j in range(layer_sizes['n_inputs']):
        for i in range(layer_sizes['n_hidden']):
            params['w'][j][i] -= learning_rate * grads['dk_dw'][j][i]
    for i in range(layer_sizes['n_hidden']):
        params['b'][i] -= learning_rate * grads['dk_db'][i]
    return params

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
            params = update_weights(params=params, grads=grads, learning_rate=learning_rate, layer_sizes=layer_sizes)
        mean_cost = sum(cost_per_instance) / len(cost_per_instance)
        cost_per_epoch.append(mean_cost)
        print(f'Epoch {epoch}, cost: {mean_cost:.5f}')
    return params

def neural_network():
    """
    Constructs and trains a neural network
    :return: The parameters (weights) of the neural network
    """
    # Define the layer sizes
    layer_sizes = {
        'n_inputs': 2,
        'n_hidden': 3,
        'n_outputs': 2
    }
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1., 1.))
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
    # Train the model and return the resulting parameters
    return optimize(params=params, X=xtrain, Y=ytrain, n_epochs=1000, learning_rate=0.01, layer_sizes=layer_sizes)

if __name__ == '__main__':
    neural_network()
