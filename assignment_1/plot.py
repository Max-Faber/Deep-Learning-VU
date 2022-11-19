import json
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def plot_learning_curve(train, val, type):
    t = np.arange(len(train))

    mean_train = train.mean(axis=1)
    mean_val = val.mean(axis=1)

    std_train = train.std(axis=1)
    std_val = val.std(axis=1)

    fig, ax = plt.subplots(1)
    ax.plot(t, mean_train, lw=2, label='Mean loss per epoch (train)', color='blue')
    ax.plot(t, mean_val, lw=2, label='Mean loss per epoch (validation)', color='orange')
    ax.fill_between(t, mean_train + std_train, mean_train - std_train, facecolor='blue', alpha=0.5)
    ax.fill_between(t, mean_val + std_val, mean_val - std_val, facecolor='orange', alpha=0.5)
    ax.set_title(f'Learning curve {type}')
    ax.legend(loc='best')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(type)
    plt.ylim([0, max(max(mean_train), max(mean_val)) + 0.2])
    plt.show()

def load_runs(dir_name):
    runs = []
    for file_name in glob(dir_name):
        with open(file_name, mode='r') as run_file:
            runs.append(json.load(run_file))
    return runs

def prepare_plots(runs, type):
    train = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['train'][type])))
    val = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['val'][type])))
    for i, run in enumerate(runs):
        train[i] = run['mean_stats']['train'][type]
        val[i] = run['mean_stats']['val'][type]
    prepared_train, prepared_val = np.zeros(shape=(len(runs), len(runs))), np.zeros(shape=(len(runs), len(runs)))
    for i in range(len(prepared_train)):
        prepared_train[i] = train[:, i]
        prepared_val[i] = val[:, i]
    return prepared_train, prepared_val

if __name__ == '__main__':
    type = 'loss_per_epoch'
    runs = load_runs('exports/batch32_lr01/*')
    prepared_train, prepared_val = prepare_plots(runs=runs, type=type)
    plot_learning_curve(prepared_train, prepared_val, type)
    pass