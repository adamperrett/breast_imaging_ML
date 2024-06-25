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


def evaluate_model(model, dataloader, criterion, inverse_standardize_targets, mean, std,
                   return_names=False, split_CC_and_MLO=True, r2_weighting_offset=0):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    if return_names:
        all_names = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for inputs, targets, _, _, file_names in tqdm(dataloader):
            nan_mask = torch.isnan(inputs)
            if torch.sum(nan_mask) > 0:
                print("Image is corrupted during evaluation", torch.sum(torch.isnan(inputs), dim=1))
            inputs[nan_mask] = 0
            inputs, targets = inputs.to(device), targets

            is_it_mlo = torch.zeros_like(torch.vstack([targets, targets])).T.float()
            if not split_CC_and_MLO:
                for i in range(len(file_names)):
                    if 'MLO' in file_names[i]:
                        is_it_mlo[i][0] += 1
                    else:
                        is_it_mlo[i][1] += 1

            outputs = model.forward(inputs.unsqueeze(1)).to('cpu')
            test_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
            test_targets_original_scale = targets.float() #inverse_standardize_targets(targets.float(), mean, std)
            loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(test_targets_original_scale.cpu().numpy())
            all_predictions.extend(test_outputs_original_scale.cpu().numpy())
            if return_names:
                all_names.extend(file_names)

    epoch_loss = running_loss / len(dataloader.dataset)
    if torch.sum(torch.isnan(torch.tensor(all_targets))) > 0:
        print("Corrupted targets")
    if torch.sum(torch.isnan(torch.tensor(all_predictions))) > 0:
        print("Corrupted predictions")
    error, stderr, conf_int = compute_error_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    r2w = r2_score(all_targets, all_predictions, sample_weight=np.array(all_targets)+r2_weighting_offset)
    if return_names:
        return epoch_loss, all_targets, all_predictions, r2, all_names
    else:
        return epoch_loss, all_targets, all_predictions, r2, r2w, error, conf_int

def compute_error_metrics(targets, pred):
    error = torch.tensor(pred) - torch.tensor(targets)
    stderr = torch.std(error)
    conf_int = stderr * 1.96
    return torch.mean(error), stderr, conf_int


def compute_target_statistics(labels):
    mean = np.mean(labels)
    std = np.std(labels)
    return mean, std

def standardize_targets(target, mean, std):
    return (target - mean) / std

def inverse_standardize_targets(target, mean, std):
    return target * std + mean


def compute_sample_weights(targets, n_bins=7, only_bins=False, minv=0, maxv=2**14):
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
    bin_counts += 1

    bin_weights = 1. / bin_counts
    bin_weights /= bin_weights.mean()

    # Assign weight to each sample based on its bin
    sample_weights = bin_weights[digitized]
    return sample_weights
