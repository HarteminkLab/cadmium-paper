
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def plot_distribution_times(data, times=[0, 7.5, 15, 30, 60, 120], 
    xlim=(0, 100), ylim=(0, 1000), bins=30):
    """
    Plot the distribution of a given data set for each time point.
    """

    fig, axs = plt.subplots(2, 3, figsize=(8,4))
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = np.concatenate([axs[0], axs[1]])
    for i in range(len(times)):
        time = times[i]
        ax = axs[i]
        current_data = data[time]
        current_data = current_data[(current_data > xlim[0]) &
                                    (current_data < xlim[1])]

        ax.hist(current_data.values, bins=bins, alpha=1., edgecolor='white', linewidth=0.5)
        ax.set_title("%.1f min" % time)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
