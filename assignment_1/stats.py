import json
import os
import numpy as np
from datetime import datetime

dir_name = 'exports'

def dump_stats_json(batch_size, learning_rate, layer_sizes, mean_stats):
    dt = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    if 'train' in mean_stats.keys() and 'loss_per_epoch' in mean_stats['train'].keys():
        mean_stats['train']['loss_per_epoch'] = mean_stats['train']['loss_per_epoch'].tolist() if type(mean_stats['train']['loss_per_epoch']) == np.ndarray else mean_stats['train']['loss_per_epoch']
    if 'train' in mean_stats.keys() and 'acc_per_epoch' in mean_stats['train'].keys():
        mean_stats['train']['acc_per_epoch'] = mean_stats['train']['acc_per_epoch'].tolist() if type(mean_stats['train']['acc_per_epoch']) == np.ndarray else mean_stats['train']['acc_per_epoch']
    if 'train' in mean_stats.keys() and 'batch_loss_per_timestep' in mean_stats['train'].keys():
        mean_stats['train']['batch_loss_per_timestep'] = mean_stats['train']['batch_loss_per_timestep'].tolist() if type(mean_stats['train']['batch_loss_per_timestep']) == np.ndarray else mean_stats['train']['batch_loss_per_timestep']
    if 'val' in mean_stats.keys() and 'loss_per_epoch' in mean_stats['val'].keys():
        mean_stats['val']['loss_per_epoch'] = mean_stats['val']['loss_per_epoch'].tolist() if type(mean_stats['val']['loss_per_epoch']) == np.ndarray else mean_stats['val']['loss_per_epoch']
    if 'val' in mean_stats.keys() and 'acc_per_epoch' in mean_stats['val'].keys():
        mean_stats['val']['acc_per_epoch'] = mean_stats['val']['acc_per_epoch'].tolist() if type(mean_stats['val']['acc_per_epoch']) == np.ndarray else mean_stats['val']['acc_per_epoch']
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(f"{dir_name}/{dt}.json", mode='w') as output_json:
        output_json.write(json.dumps({
            'datetime': dt,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'layer_sizes': layer_sizes,
            'mean_stats': mean_stats
        }, indent=4))

def get_accuracy(Y, T):
    diff = Y - T
    n_correct = diff == 0
    return sum(n_correct) / len(Y)
