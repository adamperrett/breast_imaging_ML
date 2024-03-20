import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score


def evaluate_model(model, dataloader, criterion, inverse_standardize_targets, mean, std):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets, _, _, _ in tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs.unsqueeze(1))
            test_outputs_original_scale = inverse_standardize_targets(outputs.squeeze(1), mean, std)
            test_targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
            loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(test_targets_original_scale.cpu().numpy())
            all_predictions.extend(test_outputs_original_scale.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    r2 = r2_score(all_targets, all_predictions)
    return epoch_loss, all_targets, all_predictions, r2

def compute_target_statistics(dataset):
    labels = [label for _, label, _ in dataset]
    mean = np.mean(labels)
    std = np.std(labels)
    return mean, std

def standardize_targets(target, mean, std):
    return (target - mean) / std

def inverse_standardize_targets(target, mean, std):
    return target * std + mean


def compute_sample_weights(targets, n_bins=20, only_bins=False, minv=0, maxv=2**14):
    # Discretize the target variable into bins
    if only_bins:
        bins = np.linspace(minv, maxv, n_bins)
    else:
        bins = np.linspace(min(targets), max(targets), n_bins)
    digitized = np.digitize(targets, bins)

    # Compute weight for each bin
    bin_counts = np.bincount(digitized, minlength=n_bins + 1)
    if only_bins:
        return bin_counts

    # Set a minimum count for bins
    # min_count = 1  # setting this to 1 ensures no divide by zero issue
    # bin_counts = np.maximum(bin_counts, min_count)
    bin_counts += 10

    bin_weights = 1. / bin_counts
    bin_weights /= bin_weights.sum()

    # Assign weight to each sample based on its bin
    sample_weights = bin_weights[digitized]
    return sample_weights
