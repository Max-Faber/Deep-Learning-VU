import json
import os
import numpy as np
from datetime import datetime
from plot import convert_data, load_runs

dir_name = 'exports'

def dump_stats_json(batch_size, learning_rate, layer_sizes, mean_stats):
    dt = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    if 'loss_per_epoch' in mean_stats['train'].keys() and type(mean_stats['train']['loss_per_epoch']) == np.ndarray:
        mean_stats['train']['loss_per_epoch'] = mean_stats['train']['loss_per_epoch'].tolist()
    if 'acc_per_epoch' in mean_stats['train'].keys() and type(mean_stats['train']['acc_per_epoch']) == np.ndarray:
        mean_stats['train']['acc_per_epoch'] = mean_stats['train']['acc_per_epoch'].tolist()
    if 'loss_per_batch' in mean_stats['train'].keys() and type(mean_stats['train']['loss_per_batch']) == np.ndarray:
        mean_stats['train']['loss_per_batch'] = mean_stats['train']['loss_per_batch'].tolist()
    if 'loss_per_instance' in mean_stats['train'] and type(mean_stats['train']['loss_per_instance']) == np.ndarray:
        mean_stats['train']['loss_per_instance'] = mean_stats['train']['loss_per_instance'].tolist()

    if 'loss_per_epoch' in mean_stats['val'].keys() and type(mean_stats['val']['loss_per_epoch']) == np.ndarray:
        mean_stats['val']['loss_per_epoch'] = mean_stats['val']['loss_per_epoch'].tolist()
    if 'acc_per_epoch' in mean_stats['val'].keys() and type(mean_stats['val']['acc_per_epoch']) == np.ndarray:
        mean_stats['val']['acc_per_epoch'] = mean_stats['val']['acc_per_epoch'].tolist()
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

if __name__ == '__main__':
    dirs = ['exports/sgd_lr001/*', 'exports/sgd_lr003/*', 'exports/sgd_lr01/*', 'exports/sgd_lr03/*']
    types = ['loss_per_epoch', 'acc_per_epoch']
    for dir in dirs:
        runs = load_runs(dir_name=dir)
        for t in types:
            _, converted_val = convert_data(runs=runs, type=t)
            last_run = converted_val[-1:]
            print(f'Dir: {dir}, type: {t}, mean: {last_run.mean()}, std: {last_run.std()}')
        pass
