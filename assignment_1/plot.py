import json
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def plot_learning_curve(run, type):
    t = np.arange(len(run['mean_stats']['train'][type]))

    mean_train = statistics.mean(run['mean_stats']['train'][type])
    mean_val = statistics.mean(run['mean_stats']['val'][type])
    std_train = statistics.stdev(run['mean_stats']['train'][type])
    std_val = statistics.stdev(run['mean_stats']['val'][type])

    lower_bound_train = mean_train * t - std_train * np.sqrt(t)
    upper_bound_train = mean_train * t + std_train * np.sqrt(t)
    lower_bound_val = mean_val * t - std_val * np.sqrt(t)
    upper_bound_val = mean_val * t + std_val * np.sqrt(t)

    fig, ax = plt.subplots(1)
    ax.plot(t, mean_train, lw=2, label='Mean loss per epoch (train)', color='blue')
    ax.plot(t, mean_val, lw=2, label='Mean loss per epoch (validation)', color='orange')
    ax.fill_between(t, mean_train + std_train, mean_train - std_train, facecolor='blue', alpha=0.5)
    ax.fill_between(t, mean_val + std_val, mean_val - std_val, facecolor='orange', alpha=0.5)
    ax.set_title(f'Learning curve {type}')
    ax.legend(loc='upper left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(type)
    plt.imshow()

def load_runs(dir_name):
    runs = []
    for file_name in glob(dir_name):
        with open(file_name, mode='r') as run_file:
            runs.append(json.load(run_file))
    return runs

def prepare_plots(runs, type):
    prepared_train = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['train'][type])))
    prepared_val = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['val'][type])))
    for i, run in enumerate(runs):
        prepared_train[i] = run['mean_stats']['train'][type]
        prepared_val[i] = run['mean_stats']['val'][type]
    return prepared_train, prepared_val

if __name__ == '__main__':
    type = 'loss_per_epoch'
    runs = load_runs('exports/batch32_lr01/*')
    prepared_train, prepared_val = prepare_plots(runs=runs, type=type)

    for run in runs:
        plot_learning_curve(run, 'loss_per_epoch')
        continue
    pass