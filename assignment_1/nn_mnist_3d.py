import numpy as np
from data import load_mnist
from nn_mnist_2d import init_parameters_example, update_weights, init_parameters, normalize, sigmoid
from stats import dump_cost_values_json, get_accuracy

def softmax(O):
    """

    :param O:
    :return:
    """
    exp = np.exp(O)
    sum = np.sum(exp, axis=-1, keepdims=True)
    return exp / sum

def forward_pass(X, params):
    k = np.matmul(X, params['w']) + params['b']
    h = sigmoid(k)
    o = np.matmul(h, params['v']) + params['c']
    y = softmax(o)
    cache = {
        'k': k,
        'h': h,
        'o': o,
        'y': y
    }
    return cache

def backward_pass(X, Y, params, cache):
    """

    :param X:
    :param Y:
    :param params:
    :param cache:
    :return:
    """
    # Backward pass of second layer
    dy_do = cache['y'] - Y
    dy_dc = np.copy(dy_do)
    do_dv = np.einsum('ij,ik->ijk', cache['h'], dy_do) # np.matmul(params['h'].T, dy_do) # np.matmul(dy_do.T, params['h']) # np.outer(params['h'], dy_do)
    do_dh = np.matmul(dy_do, params['v'].T) # np.matmul(params['v'], dy_do)
    # Backward pass of first layer
    dh_dk = do_dh * cache['h'] * (1. - cache['h'])
    dk_dw = np.einsum('ij,ik->ijk', X, dh_dk)
    dk_db = np.copy(dh_dk)
    grads = {
        'dy_dv': do_dv.mean(axis=0),
        'dy_dc': dy_dc.mean(axis=0),
        'dk_dw': dk_dw.mean(axis=0),
        'dk_db': dk_db.mean(axis=0)
    }
    return grads

def propagate(params, X_batch, Y_batch_one_hot, Y_batch):
    """
    Performs a forward- & backward pass and compute the cost
    :param params: Dictionary of parameters which are necessary for the forward- & backward pass
    :param X_batch: Batch of lists of input values
    :param Y_batch_one_hot: Batch of one hot vectors indicating the target classes
    :param Y_batch: Batch of integer values indicating the target classes
    :return: The cost (integer) and gradients (dictionary) which will be used to update the weights
    """
    # Perform a forward pass given x and the current parameters
    cache = forward_pass(X=X_batch, params=params)
    Y_predicted = np.argwhere(cache['y'] == np.amax(cache['y'], 1, keepdims=True))[np.arange(len(cache['y'])), 1]
    acc = get_accuracy(Y=Y_predicted, T=Y_batch)
    # Calculate the cost using the predicted probability of the target class
    probs_y = [c[np.where(y == 1)] for c, y in zip(cache['y'], Y_batch_one_hot)]
    costs = -np.log(probs_y)
    # Perform a backward pass given X, Y and the current parameters
    mean_grads = backward_pass(X=X_batch, Y=Y_batch_one_hot, params=params, cache=cache)
    return costs, mean_grads, acc

def optimize(params, X_train, Y_train, X_val, Y_val):
    """
    Trains the model using the given training data and target values for a given no. epochs and using a given learning rate
    :param params: Parameters (weights) of the model
    :param X_train: List of (training) instances containing the features to train on
    :param Y_train: List of (training) target values of the instances
    :param X_val: List of (validation) instances containing the features to validate the training on
    :param Y_val: List of (validation) target values of the instances
    :return: Dictionary mapping the name of the layer with its corresponding weights after training the model
    """
    mean_stats = {
        'train': {
            'cost_per_epoch': np.array([]),
            'acc_per_epoch': np.array([])
        },
        'val': {
            'cost_per_epoch': np.array([]),
            'acc_per_epoch': np.array([])
        }
    }
    n_samples = len(X_train)
    for epoch in range(1, n_epochs + 1):
        costs, accs = np.array([]), np.array([])
        for i in range(0, n_samples, batch_size):
            # Perform a propagation step using the training data
            X_batch_train = X_train[i:min(i + batch_size, n_samples)]
            Y_batch_train_one_hot = Y_train[i:min(i + batch_size, n_samples)]
            Y_batch_train = ytrain[i:min(i + batch_size, n_samples)]
            cost, mean_grads, acc = propagate(
                params=params,
                X_batch=X_batch_train,
                Y_batch_one_hot=Y_batch_train_one_hot,
                Y_batch=Y_batch_train
            )
            params = update_weights(params=params, grads=mean_grads, learning_rate=learning_rate)
            costs = np.append(costs, cost)
            accs = np.append(accs, acc)

            # Perform a propagation step using the validation data (without backward pass and weight update obviously)
            X_batch_val = X_val[i:min(i + batch_size, n_samples)]
            Y_batch_val_one_hot = Y_val[i:min(i + batch_size, n_samples)]
            Y_batch_val = yval[i:min(i + batch_size, n_samples)]
            # mean_cost, _, acc

        mean_cost = np.mean(costs)
        mean_acc = np.mean(accs)
        mean_stats['train']['cost_per_epoch'] = np.append(mean_stats['train']['cost_per_epoch'], mean_cost)
        mean_stats['train']['acc_per_epoch'] = np.append(mean_stats['train']['acc_per_epoch'], mean_acc)
        print(f'Epoch {epoch}, cost: {mean_cost:.5f}, accuracy: {mean_acc:.5f}')
    return params, mean_stats

def convert_one_hot(Y):
    """

    :param Y:
    :param n_classes:
    :return:
    """
    return np.eye(n_classes, dtype=int)[Y]

def neural_network():
    """

    :return:
    """
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1., 1.))
    parameters, mean_stats = optimize(
        params=params,
        X_train=xtrain_norm,
        Y_train=ytrain_one_hot,
        X_val=xval_norm,
        Y_val=yval_one_hot
    )
    return parameters, mean_stats

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.01
    n_epochs = 1
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), n_classes = load_mnist()
    # Normalize the training data
    xtrain_norm, xval_norm = normalize(xtrain), normalize(xval)
    ytrain_one_hot, yval_one_hot = convert_one_hot(Y=ytrain), convert_one_hot(Y=yval)
    del xtrain, xval
    # Define the layer sizes
    layer_sizes = {
        'n_inputs': xtrain_norm.shape[1],
        'n_hidden': 300,
        'n_outputs': n_classes
    }
    parameters, mean_stats = neural_network()
    dump_cost_values_json(
        batch_size=batch_size,
        learning_rate=learning_rate,
        layer_sizes=layer_sizes,
        mean_stats=mean_stats
    )
