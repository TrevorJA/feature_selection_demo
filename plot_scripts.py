"""
Trevor Amestoy
Cornell University

Contains basic plot functions used in the feature selection demo.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, log = False, save = False):
    """
    Takes a np.array and displays the distribution using a histogram.
    """
    sns.set(style='whitegrid', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, kde=True, stat='density', linewidth=0.5, color='cornflowerblue')
    sns.despine(left=True, bottom=True)
    if log:
        ax.set_xlabel('Log Mean Streamflow (cms)')
        nameadd = 'log'
    else:
        ax.set_xlabel('Mean Streamflow (cms)')
        ax.set_xlim([np.quantile(data, 0.01),np.quantile(data, 0.99)])
        nameadd = 'real'
    ax.set_ylabel('Density')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(f'figs/flow_distributions_{nameadd}.png', dpi = 300)
    plt.show()


################################################################################


def plot_multiple_distributions(data, N, feature_names, save = False):
    """
    Takes a np.array and plots several density distributions vertically
    for N random features in the dataset.
    """

    # Set style to white and increase font size
    sns.set(style="white", font_scale=1.2)
    # Set up the figure with a shared x and y axis label
    fig, ax = plt.subplots(N, 1, figsize=(N, 5), sharex=True, sharey=True)
    fig.text(0.5, 0.01, 'Value', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)

    # Loop through the number of features to plot
    for i in range(N):
        # Randomly select a feature column
        feature = np.random.randint(0, data.shape[1])
        # Plot the density distribution of the feature
        sns.kdeplot(data[:, feature], ax=ax[i], fill =True, linewidth=0.75, color='cornflowerblue')
        # Remove the axis boxes and set the tick labels
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_ylim([0, 1.1])
        ax[i].set_xlim([np.quantile(data, 0.02),np.quantile(data, 0.98)])
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([])
        ax[i].set_title(f'{feature_names[feature]}', fontsize=10)
    #plt.tight_layout()
    if save:
        plt.savefig(f'figs/input_data_distributions.png', dpi = 300)
    plt.show()


################################################################################


def plot_predicted_vs_actual(y_true, y_pred, save = False):
    """
    Creates a scatter plot of predicted versus actual values.
    """
    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))

    # Create the scatter plot
    sns.set(style="white", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, color='cornflowerblue', alpha=0.8, s=80)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle='--', color='gray', linewidth=1.5)

    # Set the axis labels and limits
    ax.set_xlabel('True Value', fontsize=16)
    ax.set_ylabel('Predicted Value', fontsize=16)
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)

    # Remove the axis boxes and set the tick labels
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Display the plot
    if save:
        plt.savefig(f'figs/MLP_predicted_vs_actual.png', dpi = 300)
    plt.show()


################################################################################
