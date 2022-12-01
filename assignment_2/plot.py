import numpy as np
from assignment_1.plot import plot_learning_curve

if __name__ == '__main__':
    accs_sigmoid = np.array([
        [0.0996,    0.0814, 0.1008, 0.1334, 0.094],     # Epoch 1
        [0.9472,    0.949,  0.9464, 0.9462, 0.9494],    # Epoch 2
        [0.9582,    0.959,  0.9574, 0.953,  0.9566],    # Epoch 3
        [0.9624,    0.961,  0.9614, 0.9616, 0.9616],    # Epoch 4
        [0.9646,    0.9646, 0.9652, 0.9654, 0.9656],    # Epoch 5
        [0.9652,    0.967,  0.9666, 0.9654, 0.9674],    # Epoch 6
        [0.9656,    0.9694, 0.9688, 0.9668, 0.9706],    # Epoch 7
        [0.9668,    0.97,   0.9702, 0.9684, 0.9712],    # Epoch 8
        [0.9678,    0.9716, 0.9696, 0.9688, 0.9706],    # Epoch 9
        [0.9698,    0.971,  0.9708, 0.9702, 0.972],     # Epoch 10
        [0.9696,    0.971,  0.9718, 0.9698, 0.9722],    # Epoch 11
        [0.9696,    0.9712, 0.9722, 0.9702, 0.973],     # Epoch 12
        [0.9708,    0.9718, 0.9724, 0.9706, 0.973],     # Epoch 13
        [0.9706,    0.9726, 0.9722, 0.9712, 0.9736],    # Epoch 14
        [0.971,     0.972,  0.9728, 0.9714, 0.973],     # Epoch 15
        [0.9708,    0.9724, 0.9732, 0.9722, 0.9732],    # Epoch 16
        [0.9708,    0.9726, 0.9742, 0.9724, 0.9732],    # Epoch 17
        [0.9712,    0.9726, 0.9738, 0.973,  0.9734],    # Epoch 18
        [0.9716,    0.9732, 0.9744, 0.973,  0.9736],    # Epoch 19
        [0.9716,    0.973,  0.9744, 0.973,  0.9732]     # Epoch 20
    ])
    accs_relu = np.array([
        [0.062,     0.074,  0.13,   0.1058, 0.0678],    # Epoch 1
        [0.8492,    0.8116, 0.8492, 0.801,  0.901],     # Epoch 2
        [0.8964,    0.9094, 0.7296, 0.8918, 0.938],     # Epoch 3
        [0.9258,    0.8678, 0.8924, 0.9372, 0.9508],    # Epoch 4
        [0.9338,    0.9454, 0.9204, 0.9328, 0.956],     # Epoch 5
        [0.9396,    0.947,  0.931,  0.9128, 0.9612],    # Epoch 6
        [0.9564,    0.9576, 0.9368, 0.9434, 0.9582],    # Epoch 7
        [0.9588,    0.9602, 0.9368, 0.9508, 0.961],     # Epoch 8
        [0.926,     0.9604, 0.9436, 0.9522, 0.9636],    # Epoch 9
        [0.958,     0.9554, 0.9532, 0.9564, 0.9644],    # Epoch 10
        [0.9496,    0.9616, 0.9456, 0.9556, 0.9642],    # Epoch 11
        [0.9544,    0.9622, 0.9566, 0.9548, 0.9632],    # Epoch 12
        [0.9526,    0.9606, 0.9598, 0.9576, 0.9678],    # Epoch 13
        [0.9612,    0.9622, 0.9596, 0.9598, 0.9662],    # Epoch 14
        [0.9542,    0.9632, 0.952,  0.9612, 0.9658],    # Epoch 15
        [0.9566,    0.9628, 0.9626, 0.9578, 0.9672],    # Epoch 16
        [0.9568,    0.965,  0.9606, 0.9598, 0.9662],    # Epoch 17
        [0.9636,    0.962,  0.9598, 0.9594, 0.9676],    # Epoch 18
        [0.9624,    0.9624, 0.9602, 0.9638, 0.97],      # Epoch 19
        [0.9664,    0.9636, 0.965,  0.9638, 0.9676]     # Epoch 20
    ])
    plot_learning_curve(
        title='Mean accuracy per epoch averaged over 5 runs',
        x_label='Epoch',
        y_label='Accuracy',
        vals_a=accs_sigmoid,
        label_a='Model with Sigmoid activation',
        vals_b=accs_relu,
        label_b='Model with ReLU activation'
    )
