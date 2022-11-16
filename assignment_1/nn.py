import math

def sigmoid(x):
    """
    Calculate the sigmoid function over the given input
    :param x: Non-activated neuron
    :return: Sigmoid activated neuron
    """
    return 1 / (1 + math.exp(-x))

def softmax(O):
    """
    Calculate the softmax function over all the given values
    :param O: List of non-activated neurons
    :return: List of softmax-activated neurons
    """
    exp_sum = sum([math.exp(o) for o in O])
    return [math.exp(o) / exp_sum for o in O]

def init_parameters():
    """
    Init the parameters the way as they're presented in the assignment
    :return: Dictionary containing the parameters
    """
    return {
        'w': [[1., 1., 1.], [-1., -1., -1.]],
        'b': [0., 0., 0.],
        'v': [[1., 1.], [-1., -1.], [-1., -1.]],
        'c': [0., 0.]
    }

def forward_pass(x, params, layer_sizes):
    """
    Performs a forward pass of the neural network
    :param x: List containing the input values (features)
    :param params: Dictionary mapping the layer name to its corresponding parameters (weights)
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :return: Cache dictionary mapping the necessary layer names to its corresponding parameters (weights),
    as these are necessary to perform a backward pass later
    """
    # Generate lists in which the results of the forward pass will be stored
    k = [0. for _ in range(layer_sizes['n_hidden'])]
    h = [0. for _ in range(layer_sizes['n_hidden'])]
    o = [0. for _ in range(layer_sizes['n_outputs'])]
    # Calculate results of the hidden layer (k, h)
    for j in range(layer_sizes['n_hidden']):
        for i in range(layer_sizes['n_inputs']):
            k[j] += params['w'][i][j] * x[i]
        k[j] += params['b'][j]
        h[j] = sigmoid(k[j])
    # Calculate the results of the output layer (o, y)
    for j in range(layer_sizes['n_outputs']):
        for i in range(layer_sizes['n_hidden']):
            o[j] += h[i] * params['v'][i][j]
        o[j] += params['c'][j]
    y = softmax(O=o)
    # Store the outcomes of the forward in a cache dictionary as they are necessary to perform the upcoming backward pass
    cache = {
        'k': k,
        'h': h,
        'o': o,
        'y': y
    }
    return cache

def backward_pass(x, y, params, cache, layer_sizes):
    """
    Performs a backward pass of the neural network
    :param x: List containing the input values (features)
    :param y: List containing the target class values
    :param params: Dictionary mapping the layer name to its corresponding parameters (weights)
    :param cache: Dictionary mapping layer names to its corresponding values resulting from the forward pass
    :param layer_sizes: Dictionary mapping the layers to the no. neurons in that layer
    :return: Dictionary mapping the gradient names to its corresponding values,
    which are necessary for updating the weights
    """
    # Generate lists in which the gradients will be stored
    dl_dy = [0. for _ in range(layer_sizes['n_outputs'])]
    dy_do = [0. for _ in range(layer_sizes['n_outputs'])]
    do_dc = [0. for _ in range(layer_sizes['n_outputs'])]
    do_dv = [[0. for _ in range(layer_sizes['n_outputs'])] for _ in range(layer_sizes['n_hidden'])]
    do_dh = [0. for _ in range(layer_sizes['n_hidden'])]
    dh_dk = [0. for _ in range(layer_sizes['n_hidden'])]
    dk_db = [0. for _ in range(layer_sizes['n_hidden'])]
    dk_dw = [[0. for _ in range(layer_sizes['n_hidden'])] for _ in range(layer_sizes['n_inputs'])]
    # Calculate gradients of the output layer (dl_dy, dy_do, dy_dc)
    dl_dy[y] = -1. / cache['y'][y]
    for j in range(layer_sizes['n_outputs']):
        for i in range(layer_sizes['n_outputs']):
            if i == j:
                dy_do[j] += dl_dy[i] * cache['y'][j] * (1. - cache['y'][i])
            else:
                dy_do[j] += dl_dy[i] * -cache['y'][j] * cache['y'][i]
        do_dc[j] = dy_do[j]
    # Calculate gradients of the hidden layer (do_dv, do_dh)
    for j in range(layer_sizes['n_outputs']):
        for i in range(layer_sizes['n_hidden']):
            do_dv[i][j] = dy_do[j] * cache['h'][i]
            do_dh[i] += dy_do[j] * params['v'][i][j]
    # Calculate gradients of the input layer (dh_dk, dk_dw, dk_db)
    for j in range(layer_sizes['n_hidden']):
        dh_dk[j] = do_dh[j] * cache['h'][j] * (1. - cache['h'][j])
        for i in range(layer_sizes['n_inputs']):
            dk_dw[i][j] = dh_dk[j] * x[i]
        dk_db[j] = dh_dk[j]
    # Store the gradients which are necessary to update the weights in a dictionary and return it
    grads = {
        'do_dv': do_dv,
        'do_dc': do_dc,
        'dk_dw': dk_dw,
        'dk_db': dk_db
    }
    return grads

def neural_network():
    """
    Constructs a neural network and performs 1 forward- & backward pass
    :return: None
    """
    # Desired architecture for the neural network
    layer_sizes = {
        'n_inputs': 2,
        'n_hidden': 3,
        'n_outputs': 2
    }
    # Test inputs and output given in the assignment
    x = [1., -1.]
    y = 0 # Equivalent to [1, 0] in one-hot encoding
    params = init_parameters()
    cache = forward_pass(x=x, params=params, layer_sizes=layer_sizes)
    grads = backward_pass(x=x, y=y, params=params, cache=cache, layer_sizes=layer_sizes)
    print(f"do_dv: {grads['do_dv']}")
    print(f"do_dc: {grads['do_dc']}")
    print(f"dk_dw: {grads['dk_dw']}")
    print(f"dk_db: {grads['dk_db']}")

if __name__ == '__main__':
    neural_network()
