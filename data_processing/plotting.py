import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, task_name="Task", save_location=None, return_figure=True):
    cm = confusion_matrix(y_true, y_pred)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix for {task_name}")

    # Save the figure if a save location is provided
    if save_location:
        fig.savefig(f"{save_location}/confusion_matrix_{task_name}.png", bbox_inches='tight', format='png')

    # Return or show the figure
    if return_figure:
        return fig
    else:
        plt.show()



def plot_auc_curves(all_labels, all_preds, data_subset, task_names=None, save_location=None, return_figure=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    title = "ROC Curves for {}".format(data_subset)

    for i in range(len(all_preds)):  # Iterate over tasks
        fpr, tpr, _ = roc_curve(all_labels[i], all_preds[i])
        if len(set(all_labels[i])) > 1:  # Ensure both 0 and 1 exist
            auc_score = roc_auc_score(all_labels[i], all_preds[i])
        else:
            print(f"Skipping AUC for Task {i + 1} {data_subset}: Only one class present in validation labels.")
            auc_score = 0.0

        if task_names:
            label = f"{task_names[i]} (AUC = {auc_score:.3f})"
        else:
            label = f"Task {i + 1} (AUC = {auc_score:.3f})"

        ax.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if save_location:
        fig.savefig(f"{save_location}/{title}.png", bbox_inches='tight', format='png')
    if return_figure:
        return fig
    else:
        plt.show()


def plot_scatter(true_values, pred_values, title, save_location=None, return_figure=True, manufacturers=None):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes objects

    # Scatter plot: color the points based on labels if provided
    if manufacturers is not None:
        unique_labels = np.unique(manufacturers)

        # Plot each label (manufacturer) separately
        for label in unique_labels:
            mask = np.array(manufacturers) == label
            ax.scatter(np.array(true_values)[mask], np.array(pred_values)[mask], label=label, alpha=0.5)

        # Add legend for manufacturers
        ax.legend(title="Manufacturer")
    else:
        # Simple scatter plot without labels
        ax.scatter(true_values, pred_values, alpha=0.5)

    # Plot the reference line y=x
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--', lw=2, color='red')

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    if save_location:
        fig.savefig(f"{save_location}/{title}.png", bbox_inches='tight', format='png')
    if return_figure:
        return fig
    else:
        plt.show()

def plot_error_vs_vas(true_values, pred_values, title, save_location=None, min_e=-40, max_e=40, manufacturers=None):
    plt.figure(figsize=(10, 6))
    errors = np.array(true_values) - np.array(pred_values)

    # Scatter plot: Color by manufacturers (labels) if provided
    if manufacturers is not None:
        # Convert manufacturers (or other labels) to numeric if needed
        unique_labels = np.unique(manufacturers)

        # Plot each label (manufacturer) separately
        for label in unique_labels:
            mask = np.array(manufacturers) == label
            plt.scatter(np.array(true_values)[mask], np.array(errors)[mask], label=label, alpha=0.5)

        # Add legend for manufacturers
        plt.legend(title="Manufacturer")
    else:
        # Simple scatter plot without labels
        plt.scatter(true_values, errors, alpha=0.5)

    # Set plot labels and limits
    plt.xlabel('VAS')
    plt.ylabel('Error')
    plt.ylim([min_e, max_e])
    if np.sum([e < min_e or e > max_e for e in errors]):
        print("An error was out of bounds")
        print("Max:", np.max(errors), "Min:", np.min(errors))
    plt.title(title)
    if save_location:
        plt.savefig(f"{save_location}/{title}.png", bbox_inches='tight', format='png')
    else:
        plt.show()

def plot_error_distribution(true_values, pred_values, title, save_location=None, manufacturers=None):
    plt.figure(figsize=(10, 6))
    errors = np.array(true_values) - np.array(pred_values)
    err = np.mean(errors)
    stderr = np.std(errors)

    # Prepare data for plotting
    data = {'Errors': errors}

    # If labels are provided, add them to the data
    if manufacturers is not None:
        data['Manufacturers'] = manufacturers

    # Convert data to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(data)

    # Plot the histogram with seaborn, using the 'hue' parameter for labels
    sns.histplot(df, x='Errors', hue='Manufacturers', bins=20, kde=False, multiple='stack', palette='viridis')

    # Set plot labels and title
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(f"{title} - mean error: {err:.2f} +/- {stderr:.2f}")

    # Save the plot or show it
    if save_location:
        plt.savefig(f"{save_location}/{title}.png", bbox_inches='tight', format='png')
    else:
        plt.show()