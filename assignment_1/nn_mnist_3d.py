import random
import numpy as np
from data import load_mnist
from nn_mnist_2d import init_parameters_example, update_weights, init_parameters, normalize, sigmoid
from stats import dump_stats_json, get_accuracy

def softmax(O):
    """
    Applies softmax function to nested numpy array
    :param O: Non activated neurons
    :return: Activated neurons
    """
    exp = np.exp(O)
    sum = np.sum(exp, axis=-1, keepdims=True)
    return exp / sum

def forward_pass(X, params):
    """
    3D Version of the forward pass
    :param X: Input values
    :param params: Required parameters for the forward pass
    :return: Required values for the rest of the training loop
    """
    # Forward pass of first layer
    k = np.matmul(X, params['w']) + params['b']
    h = sigmoid(k)
    # Forward pass of second layer
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
    Performs 3D version of the backward pass
    :param X: Input values
    :param Y: Output values (one-hot encoded)
    :param params: Required parameters for the backward pass
    :param cache: Weights
    :return: Averaged derivatives after performing the backward pass
    """
    # Backward pass of second layer
    dy_do = cache['y'] - Y
    dy_dc = np.copy(dy_do)
    do_dv = np.einsum('ij,ik->ijk', cache['h'], dy_do)
    do_dh = np.matmul(dy_do, params['v'].T)
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

def propagate(params, X_batch, Y_batch_one_hot, Y_batch, perform_backward=True):
    """
    Performs a forward- & backward pass and compute the loss
    :param params: Dictionary of parameters which are necessary for the forward- & backward pass
    :param X_batch: Batch of lists of input values
    :param Y_batch_one_hot: Batch of one hot vectors indicating the target classes
    :param Y_batch: Batch of integer values indicating the target classes
    :return: The loss (integer) and gradients (dictionary) which will be used to update the weights
    """
    # Perform a forward pass given x and the current parameters
    cache = forward_pass(X=X_batch, params=params)
    Y_predicted = np.argwhere(cache['y'] == np.amax(cache['y'], 1, keepdims=True))[np.arange(len(cache['y'])), 1]
    acc = get_accuracy(Y=Y_predicted, T=Y_batch)
    # Calculate the loss using the predicted probability of the target class
    probs_y = [c[np.where(y == 1)] for c, y in zip(cache['y'], Y_batch_one_hot)]
    losses = -np.log(probs_y)
    if not perform_backward:
        return losses, None, acc
    # Perform a backward pass given X, Y and the current parameters
    mean_grads = backward_pass(X=X_batch, Y=Y_batch_one_hot, params=params, cache=cache)
    return losses, mean_grads, acc

def optimize(params, X_train, Y_train, Y_train_one_hot, X_val, Y_val, Y_val_one_hot):
    """
    Trains the model using the given training data and target values for a given no. epochs and using a given learning rate
    Collects stat data for the plots in the report
    :param params: Parameters (weights) of the model
    :param X_train: List of (training) instances containing the features to train on
    :param Y_train: List of (training) target values of the instances
    :param X_val: List of (validation) instances containing the features to validate the training on
    :param Y_val: List of (validation) target values of the instances
    :return: Dictionary mapping the name of the layer with its corresponding weights after training the model
    """
    mean_stats = {
        'train': {
            'loss_per_epoch': np.array([]),
            'acc_per_epoch': np.array([]),
            'loss_per_batch': np.array([]),
            'loss_per_instance': np.array([])
        },
        'val': {
            'loss_per_epoch': np.array([]),
            'acc_per_epoch': np.array([])
        }
    }
    n_samples_train = len(X_train)
    n_samples_val = len(X_val)
    for epoch in range(1, n_epochs + 1):
        losses_train, accs_train, losses_val, accs_val = np.array([]), np.array([]), np.array([]), np.array([])
        # Shuffle the data (Gradient Descent)
        zipped = list(zip(X_train, Y_train, Y_train_one_hot))
        random.shuffle(zipped)
        X_train, Y_train, Y_train_one_hot = zip(*zipped)
        for i in range(0, n_samples_train, batch_size):
            # Perform a propagation step using the training data
            # Grab the batch
            X_batch_train = X_train[i:min(i + batch_size, n_samples_train)]
            Y_batch_train_one_hot = Y_train_one_hot[i:min(i + batch_size, n_samples_train)]
            Y_batch_train = Y_train[i:min(i + batch_size, n_samples_train)]
            losses_batch_train, mean_grads, acc_train = propagate(
                params=params,
                X_batch=X_batch_train,
                Y_batch_one_hot=Y_batch_train_one_hot,
                Y_batch=Y_batch_train
            )
            params = update_weights(params=params, grads=mean_grads, learning_rate=learning_rate)
            losses_train = np.append(losses_train, losses_batch_train)
            accs_train = np.append(accs_train, acc_train)
            mean_batch_loss = losses_batch_train.sum() / losses_batch_train.size
            mean_stats['train']['loss_per_batch'] = np.append(mean_stats['train']['loss_per_batch'], mean_batch_loss)
            mean_stats['train']['loss_per_instance'] = np.append(mean_stats['train']['loss_per_instance'], losses_batch_train)
        # Validate the training using the validation data
        for i in range(0, n_samples_val, batch_size):
            # Perform a propagation step using the validation data (without backward pass and weight update obviously)
            X_batch_val = X_val[i:min(i + batch_size, n_samples_val)]
            Y_batch_val_one_hot = Y_val_one_hot[i:min(i + batch_size, n_samples_val)]
            Y_batch_val = Y_val[i:min(i + batch_size, n_samples_val)]
            loss_val, _, acc_val = propagate(
                params=params,
                X_batch=X_batch_val,
                Y_batch_one_hot=Y_batch_val_one_hot,
                Y_batch=Y_batch_val,
                perform_backward=False
            )
            losses_val = np.append(losses_val, loss_val)
            accs_val = np.append(accs_val, acc_val)
        mean_loss_train = np.mean(losses_train)
        mean_acc_train = np.mean(accs_train)
        mean_loss_val = np.mean(losses_val)
        mean_acc_val = np.mean(accs_val)
        mean_stats['train']['loss_per_epoch'] = np.append(mean_stats['train']['loss_per_epoch'], mean_loss_train)
        mean_stats['train']['acc_per_epoch'] = np.append(mean_stats['train']['acc_per_epoch'], mean_acc_train)
        mean_stats['val']['loss_per_epoch'] = np.append(mean_stats['val']['loss_per_epoch'], mean_loss_val)
        mean_stats['val']['acc_per_epoch'] = np.append(mean_stats['val']['acc_per_epoch'], mean_acc_val)
        print(f'Epoch {epoch}, train loss: {mean_loss_train:.3f}, train accuracy: {mean_acc_train:.3f}, val loss: {mean_loss_val:.3f}, val accuracy: {mean_acc_val:.3f}')
    return params, mean_stats

def convert_one_hot(Y):
    """
    Convert int representation to one-hot encoding
    :param Y: Int represented target classes
    :param n_classes: No. classes
    :return: One-hot encoded target classes
    """
    return np.eye(n_classes, dtype=int)[Y]

def neural_network():
    """
    Constructs the neural network
    :return: Parameters after training, statistics for generating plots for the report
    """
    # Initialize the weights using the layer sizes and weight range
    params = init_parameters(layer_sizes=layer_sizes, weight_range=(-1., 1.))
    parameters, mean_stats = optimize(
        params=params,
        X_train=xtrain_norm,
        Y_train=ytrain,
        Y_train_one_hot=ytrain_one_hot,
        X_val=xval_norm,
        Y_val=yval,
        Y_val_one_hot=yval_one_hot
    )
    return parameters, mean_stats

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.03
    n_epochs = 5
    # Load the data to train on
    (xtrain, ytrain), (xval, yval), n_classes = load_mnist(final=True)
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
    dump_stats_json(
        batch_size=batch_size,
        learning_rate=learning_rate,
        layer_sizes=layer_sizes,
        mean_stats=mean_stats
    )
