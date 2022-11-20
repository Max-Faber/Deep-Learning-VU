import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from glob import glob

def plot_learning_curve(title, x_label, y_label, vals_a, label_a, vals_b=None, label_b=None, title_font_size=10):
    t = np.arange(len(vals_a)) + 1

    fig, ax = plt.subplots(1)
    ax.set_title(title, fontsize=title_font_size)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    datapoints_train_mean = vals_a.mean(axis=1)
    datapoints_train_std = vals_a.std(axis=1)
    datapoints_train = datapoints_train_mean + datapoints_train_std
    ax.fill_between(t, datapoints_train_mean + datapoints_train_std, datapoints_train_mean - datapoints_train_std, facecolor='blue', alpha=0.5)
    ax.plot(t, datapoints_train, lw=2, label=label_a, color='blue')

    if vals_b is not None:
        mean_val = vals_b.mean(axis=1)
        std_val = vals_b.std(axis=1)
        ax.plot(t, mean_val, lw=2, label=label_b, color='orange')
        ax.fill_between(t, mean_val + std_val, mean_val - std_val, facecolor='orange', alpha=0.5)
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def scatter_plot(X, title, x_label, y_label):
    t = np.arange(1, len(X) + 1)
    plt.scatter(t, X, s=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, fontsize=10)
    plt.show()

def load_runs(dir_name):
    runs = []
    for file_name in glob(dir_name):
        if not os.path.isfile(file_name):
            continue
        with open(file_name, mode='r') as run_file:
            runs.append(json.load(run_file))
    return runs

def convert_data(runs, type):
    train = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['train'][type])))
    converted_val = None
    if 'val' in runs[0]['mean_stats'].keys():
        val = np.zeros(shape=(len(runs), len(runs[0]['mean_stats']['val'][type])))
        converted_val = np.zeros(shape=(len(runs[0]['mean_stats']['val'][type]), len(runs)))
    converted_train = np.zeros(shape=(len(runs[0]['mean_stats']['train'][type]), len(runs)))
    for i, run in enumerate(runs):
        train[i] = run['mean_stats']['train'][type]
        if 'val' in runs[0]['mean_stats'].keys():
            val[i] = run['mean_stats']['val'][type]
    for i in range(len(converted_train)):
        converted_train[i] = train[:, i]
        if 'val' in runs[0]['mean_stats'].keys():
            converted_val[i] = val[:, i]
    return converted_train, converted_val

if __name__ == '__main__':
    # runs = load_runs('exports/batch32_lr01/loss_per_instance_and_batch/*')
    # scatter_plot(
    #     X=np.array(runs[0]['mean_stats']['train']['loss_per_batch']),
    #     title='Mean loss per batch on MNIST dataset (LR=0.01, batch size=32, epochs=5)',
    #     x_label='Timestep',
    #     y_label='Cross-entropy loss'
    # )

    # prepared_train, prepared_val = prepare_plots(
    #     runs=load_runs('exports/batch32_lr01/*'),
    #     type='loss_per_epoch'
    # )
    # plot_learning_curve(
    #     title='Mean loss per epoch averaged over 5 runs (LR=0.01, batch size=32, epochs=5)',
    #     x_label='Epoch',
    #     y_label='Cross-entropy loss',
    #     vals_a=prepared_train,
    #     label_a='Mean loss per epoch (train)',
    #     vals_b=prepared_val,
    #     label_b='Mean loss per epoch (validation)'
    # )

    # prepared_train_acc, prepared_val_acc = convert_data(
    #     runs=load_runs('exports/batch32_lr01/*'),
    #     type='acc_per_epoch'
    # )
    # plot_learning_curve(
    #     title='Mean accuracy per epoch averaged over 5 runs (LR=0.01, batch size=32, epochs=5)',
    #     x_label='Epoch',
    #     y_label='Accuracy',
    #     vals_a=prepared_train_acc,
    #     label_a='Mean accuracy per epoch (train)',
    #     vals_b=prepared_val_acc,
    #     label_b='Mean accuracy per epoch (validation)'
    # )

    prepared_train_acc, _ = convert_data(
        runs=load_runs('exports/synth_lr01/*'),
        type='loss_per_epoch'
    )
    plot_learning_curve(
        title='Mean loss per epoch averaged over 5 runs (LR=0.01, epochs=100)',
        x_label='Epoch',
        y_label='Cross-entropy loss',
        vals_a=prepared_train_acc,
        label_a='Mean accuracy per epoch (train)'
    )
