import random
import math
from data import load_synth
from nn_plain_python import forward_pass, backward_pass
from nn_mnist_2d import normalize
from stats import dump_stats_json

def init_parameters(layer_sizes, weight_range):
    """
    Initializes the parameters of the neural network using the given layer sizes and weight range
    :param layer_sizes: Dictionary mapping the size (no. neurons) per layer
    :param weight_range: Tuple indicating the min. and max. range of the weights which are initialized randomly
    :return: Dictionary mapping the layer names to its corresponding parameters (weights)
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
    predicted_class = cache['y'].index(max(cache['y']))
    cost = -math.log(prob_y, math.e)
    # Perform a backward pass given x, y and the current parameters
    grads = backward_pass(x=x, y=y, params=params, cache=cache, layer_sizes=layer_sizes)
    return cost, grads, predicted_class

def update_weights(params, grads, learning_rate, layer_sizes):
    """
    Updates the weights of the neural network
    :param params: Parameters (weights) of the model
    :param grads: Necessary weights to update the weights
    :param learning_rate: Learning rate for updating the weights
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :return: Dictionary containing the updated parameters
    """
    for j in range(layer_sizes['n_hidden']):
        for i in range(layer_sizes['n_outputs']):
            params['v'][j][i] -= learning_rate * grads['do_dv'][j][i]
    for i in range(layer_sizes['n_outputs']):
        params['c'][i] -= learning_rate * grads['do_dc'][i]
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
    mean_loss_per_epoch = []
    mean_acc_per_epoch = []
    for epoch in range(1, n_epochs + 1):
        loss_per_instance = []
        n_correct = 0
        for x, y in zip(X, Y):
            loss, grads, predicted_class = propagate(params=params, x=x, y=y, layer_sizes=layer_sizes)
            loss_per_instance.append(loss)
            n_correct += 1 if predicted_class == y else 0
            params = update_weights(params=params, grads=grads, learning_rate=learning_rate, layer_sizes=layer_sizes)
        mean_loss = sum(loss_per_instance) / len(loss_per_instance)
        mean_loss_per_epoch.append(mean_loss)
        mean_acc_per_epoch.append(n_correct / len(X))
        print(f'Epoch {epoch}, cost: {mean_loss:.5f}')
    mean_stats = {
        'train': {
            'loss_per_epoch': mean_loss_per_epoch,
            'acc_per_epoch': mean_acc_per_epoch
        }
    }
    return params, mean_stats

def neural_network():
    """
    Constructs and trains a neural network
    :return: The parameters (weights) of the neural network
    """
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1, 1.))
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
    xtrain_normalized = normalize(X=xtrain)
    del xtrain
    # Train the model and return the resulting parameters
    return optimize(params=params, X=xtrain_normalized, Y=ytrain, n_epochs=100, learning_rate=learning_rate, layer_sizes=layer_sizes)

if __name__ == '__main__':
    learning_rate = 0.01
    # Define the layer sizes
    layer_sizes = {
        'n_inputs': 2,
        'n_hidden': 3,
        'n_outputs': 2
    }
    parameters, mean_stats = neural_network()
    dump_stats_json(batch_size=1, learning_rate=learning_rate, layer_sizes=layer_sizes, mean_stats=mean_stats)
