"""
Trevor Amestoy
Cornell University

Contains basic plot functions used in the feature selection demo.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_vs_actual(p, o):

    fig, ax = plt.subplots()

    ax.scatter(p, o, color = 'slategrey')
    plt.title('Comparison of predictions vs actual values')
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    plt.savefig(f'nn_prediction_vs_actual.png', dpi =250)
    plt.show()
    return
