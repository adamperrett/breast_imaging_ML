import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score


def plot_scatter(true_values, pred_values, title, save_location=None, return_figure=True):
    fig, ax = plt.subplots(figsize=(10, 6))  # Use plt.subplots() to get the figure and axes objects
    ax.scatter(true_values, pred_values, alpha=0.5)
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    if save_location:
        fig.savefig(f"{save_location}/{title}.png", bbox_inches='tight', format='png')
    if return_figure:
        return fig
    else:
        plt.show()

def plot_error_vs_vas(true_values, pred_values, title, save_location=None, min_e=-40, max_e=40):
    plt.figure(figsize=(10, 6))
    errors = np.array(true_values) - np.array(pred_values)
    plt.scatter(true_values, errors, alpha=0.5)
    plt.xlabel('VAS')
    plt.ylabel('Error')
    plt.ylim([min_e, max_e])
    if np.sum([e < min_e or e > max_e for e in errors]):
        print("An error was out of bounds")
        print("Max:", np.max(errors), "Min:", np.min(errors))
    plt.title(title)
    if save_location:
        plt.savefig(save_location+"/{}.png".format(title), bbox_inches='tight',
                    # dpi=200,
                    format='png')
    else:
        plt.show()

def plot_error_distribution(true_values, pred_values, title, save_location=None):
    plt.figure(figsize=(10, 6))
    errors = np.array(true_values) - np.array(pred_values)
    sns.histplot(errors, bins=20, kde=False)
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(title)
    if save_location:
        plt.savefig(save_location+"/{}.png".format(title), bbox_inches='tight',
                    # dpi=200,
                    format='png')
    else:
        plt.show()