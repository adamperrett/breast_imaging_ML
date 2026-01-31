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
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import accuracy_score


def evaluate_CRUK(model, dataloader, criterion, subtype_mapping, train_indexes):
    model.eval()
    running_loss = 0.0
    num_classes = len(subtype_mapping)
    all_preds = [[] for _ in range(num_classes)]
    all_labels = [[] for _ in range(num_classes)]

    with torch.no_grad():
        aucs = [BinaryAUROC() for _ in range(num_classes)]
        for image_data, CRUK_data in tqdm(dataloader):

            CRUK_data = CRUK_data.T[train_indexes]

            outputs = model.forward(image_data).to('cuda')
            # test_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
            # test_targets_original_scale = targets.float() #inverse_standardize_targets(targets.float(), mean, std)
            # loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            # running_loss += loss.item() * inputs.size(0)
            split_losses = []
            for i, t in enumerate(CRUK_data):
                # loss = criterion(outputs[:, [2*i, 2*i+1]], torch.vstack([1-t.to(torch.float32), t.to(torch.float32)]).T)
                loss = criterion(outputs[:, [2*i, 2*i+1]], t.T.long())
                split_losses.append(loss)

            total_loss = torch.sum(torch.stack(split_losses), dim=1).to('cpu')
            running_loss += total_loss
            # train_loss += weighted_loss.item() * inputs.size(0)

            # print("Before scaling loss\nCurrent GPU mem usage is", torch.cuda.memory_allocated() / (1024 ** 2))

            for i in range(num_classes):
                sm = torch.softmax(outputs[:, [2*i, 2*i+1]], dim=1)
                prediction = torch.max(sm, dim=1)[1]
                all_preds[i].extend(prediction.cpu().numpy())
                all_labels[i].extend(CRUK_data[i].cpu().numpy())

            for i, auc in enumerate(aucs):
                auc.update(prediction, CRUK_data[i])

            # all_targets.extend(test_targets_original_scale.cpu().numpy())
            # all_predictions.extend(test_outputs_original_scale.cpu().numpy())
            # all_manufacturers.extend(manu)
            # if return_names:
            #     patients.extend(patient)
            #     timepoints.extend(timepoint)

    epoch_loss = running_loss / len(dataloader.dataset)
    auc_scores = torch.stack([auc.compute() for auc in aucs])
    accuracies = []
    full_binary = []
    for i in range(num_classes):
        preds_binary = (np.array(all_preds[i]) > 0.5).astype(int)  # Apply threshold for binary classification
        acc = accuracy_score(all_labels[i], preds_binary)
        accuracies.append(acc)
        full_binary.append(preds_binary)
    return epoch_loss, all_labels, all_preds, torch.tensor(accuracies), full_binary, auc_scores


def evaluate_recurrence(model, dataloader, criterion, inverse_standardize_targets, mean, std,
                    num_manufacturers, manufacturer_mapping, recurrence_mapping,
                    return_names=False, split_CC_and_MLO=True, r2_weighting_offset=0):
    model.eval()
    running_loss = 0.0
    num_classes = len(recurrence_mapping)
    all_preds = [[] for _ in range(num_classes)]
    all_labels = [[] for _ in range(num_classes)]
    all_manufacturers = []
    if return_names:
        patients = []
        timepoints = []

    with torch.no_grad():
        aucs = [BinaryAUROC() for _ in range(num_classes)]
        for image_data, recurrence_data in tqdm(dataloader):

            recurrence_data = torch.stack(recurrence_data).to('cuda')

            outputs = model.forward(image_data, manufacturer_mapping).to('cuda')
            # test_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
            # test_targets_original_scale = targets.float() #inverse_standardize_targets(targets.float(), mean, std)
            # loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            # running_loss += loss.item() * inputs.size(0)
            split_losses = []
            for i, t in enumerate(recurrence_data):
                loss = criterion(outputs[:, [2*i, 2*i+1]], torch.vstack([1-t.to(torch.float32), t.to(torch.float32)]).T)
                split_losses.append(loss)

            total_loss = torch.sum(torch.stack(split_losses), dim=1).to('cpu')
            running_loss += total_loss
            # train_loss += weighted_loss.item() * inputs.size(0)

            print("Before scaling loss\nCurrent GPU mem usage is", torch.cuda.memory_allocated() / (1024 ** 2))
            for i, auc in enumerate(aucs):
                auc.update(outputs[:, i], recurrence_data[i])

            for i in range(num_classes):
                all_preds[i].extend(outputs[:, i].cpu().numpy())
                all_labels[i].extend(recurrence_data[i].cpu().numpy())

            # all_targets.extend(test_targets_original_scale.cpu().numpy())
            # all_predictions.extend(test_outputs_original_scale.cpu().numpy())
            # all_manufacturers.extend(manu)
            # if return_names:
            #     patients.extend(patient)
            #     timepoints.extend(timepoint)

    epoch_loss = running_loss / len(dataloader.dataset)
    auc_scores = torch.stack([auc.compute() for auc in aucs])
    accuracies = []
    full_binary = []
    for i in range(num_classes):
        preds_binary = (np.array(all_preds[i]) > 0.5).astype(int)  # Apply threshold for binary classification
        acc = accuracy_score(all_labels[i], preds_binary)
        accuracies.append(acc)
        full_binary.append(preds_binary)
    if return_names:
        return epoch_loss, all_labels, all_preds, torch.tensor(accuracies), full_binary, auc_scores, all_manufacturers, \
               patients, timepoints
    else:
        return epoch_loss, all_labels, all_preds, torch.tensor(accuracies), full_binary, auc_scores, all_manufacturers


def evaluate_medici(model, dataloader, criterion, inverse_standardize_targets, mean, std,
                    num_manufacturers, manufacturer_mapping,
                    return_names=False, split_CC_and_MLO=True, r2_weighting_offset=0):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_manufacturers = []
    if return_names:
        patients = []
        timepoints = []

    with torch.no_grad():
        for inputs, targets, timepoint, patient, manu, views in tqdm(dataloader):
            nan_mask = torch.isnan(inputs)
            if torch.sum(nan_mask) > 0:
                print("Image is corrupted during evaluation", torch.sum(torch.isnan(inputs), dim=1))
            inputs[nan_mask] = 0
            inputs, targets = inputs.cuda(), targets

            is_it_mlo = torch.zeros([len(views[0]), len(views), 2]).float()
            if not split_CC_and_MLO:
                for i in range(len(views[0])):
                    for j in range(len(views)):
                        if 'MLO' in views[j][i]:
                            is_it_mlo[i][j][0] += 1
                        else:
                            is_it_mlo[i][j][1] += 1
            manufacturer = torch.zeros([len(views[0]), len(views), num_manufacturers]).float().to('cuda')
            for j in range(len(manufacturer[0])):
                for i in range(len(manufacturer)):
                    manufacturer[i][j] += manufacturer_mapping[manu[i]]

            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo.cuda(), manufacturer).to('cpu')
            test_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
            test_targets_original_scale = targets.float().to('cpu') #inverse_standardize_targets(targets.float(), mean, std)
            loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(test_targets_original_scale.cpu().numpy())
            all_predictions.extend(test_outputs_original_scale.cpu().numpy())
            all_manufacturers.extend(manu)
            if return_names:
                patients.extend(patient)
                timepoints.extend(timepoint)

    epoch_loss = running_loss / len(dataloader.dataset)
    if torch.sum(torch.isnan(torch.tensor(all_targets))) > 0:
        print("Corrupted targets")
    if torch.sum(torch.isnan(torch.tensor(all_predictions))) > 0:
        print("Corrupted predictions")
    error, stderr, conf_int = compute_error_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    r2w = r2_score(all_targets, all_predictions, sample_weight=np.array(all_targets)+r2_weighting_offset)
    if return_names:
        return epoch_loss, all_targets, all_predictions, r2, r2w, error, conf_int, all_manufacturers, \
               patients, timepoints
    else:
        return epoch_loss, all_targets, all_predictions, r2, r2w, error, conf_int, all_manufacturers

def evaluate_mosaic(model, dataloader, criterion, inverse_standardize_targets, mean, std,
                   return_names=False, split_CC_and_MLO=True, r2_weighting_offset=0):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    if return_names:
        all_names = []

    with torch.no_grad():
        for inputs, targets, _, _, file_names in tqdm(dataloader):
            nan_mask = torch.isnan(inputs)
            if torch.sum(nan_mask) > 0:
                print("Image is corrupted during evaluation", torch.sum(torch.isnan(inputs), dim=1))
            inputs[nan_mask] = 0
            inputs, targets = inputs.cuda(), targets

            is_it_mlo = torch.zeros([len(file_names[0]), len(file_names), 2]).float()
            if not split_CC_and_MLO:
                for i in range(len(file_names[0])):
                    for j in range(len(file_names)):
                        if 'MLO' in file_names[j][i]:
                            is_it_mlo[i][j][0] += 1
                        else:
                            is_it_mlo[i][j][1] += 1

            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo.cuda()).to('cpu')
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


def evaluate_model(model, dataloader, criterion, inverse_standardize_targets, mean, std,
                   return_names=False, split_CC_and_MLO=True, r2_weighting_offset=0):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    if return_names:
        all_names = []

    with torch.no_grad():
        for inputs, targets, _, _, file_names in tqdm(dataloader):
            nan_mask = torch.isnan(inputs)
            if torch.sum(nan_mask) > 0:
                print("Image is corrupted during evaluation", torch.sum(torch.isnan(inputs), dim=1))
            inputs[nan_mask] = 0
            inputs, targets = inputs.cuda(), targets

            is_it_mlo = torch.zeros_like(torch.vstack([targets, targets])).T.float()
            if not split_CC_and_MLO:
                for i in range(len(file_names)):
                    if 'MLO' in file_names[i]:
                        is_it_mlo[i][0] += 1
                    else:
                        is_it_mlo[i][1] += 1

            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo.cuda()).to('cpu')
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
    # labels = [label for _, label, _, _, _, _ in dataset]
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


def bland_altman_plot(data1, data2, *args, **kwargs):
    """
    Create a Bland-Altman plot.

    Parameters:
    - data1, data2: The two sets of data to be compared.
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel('Mean of Target and Prediction')
    plt.ylabel('Difference (Error)')
    plt.title('Bland-Altman Plot')


def bland_altman_plot_multiple(title, data1, data2_dict, mean_plotting=True, font_size=16, *args, **kwargs):
    """
    Create a Bland-Altman plot for multiple comparisons with different colors.

    Parameters:
    - data1: The reference data (e.g., targets).
    - data2_dict: A dictionary where keys are labels and values are the data to compare (e.g., predictions, pvas).
    """
    data1 = np.asarray(data1)

    plt.figure(figsize=(15, 9))

    # Set font size for all text elements
    plt.rcParams.update({'font.size': font_size})

    colors = plt.cm.get_cmap('tab10', len(data2_dict))  # Use a colormap with a different color for each item

    print("Evaluating", title)
    for i, (label, data2) in enumerate(data2_dict.items()):
        data2 = np.asarray(data2)
        diff = data2 - data1
        if mean_plotting:
            mean = np.mean([data1, data2], axis=0)
        else:
            mean = data1
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff, label=label, color=colors(i), *args, **kwargs)
        plt.axhline(md, color=colors(i), linestyle='-')
        plt.axhline(md + 1.96 * sd, color=colors(i), linestyle='--')
        plt.axhline(md - 1.96 * sd, color=colors(i), linestyle='--')
        print("For", label, "\tmean:", md, "\tCI:", 1.96 * sd, "\tMSE:", np.mean(np.square(diff)))
    if mean_plotting:
        plt.xlabel('Mean of Target and Prediction')
    else:
        plt.xlabel('Target Value')
    plt.ylabel('Error (Predicted - Target)')
    plt.ylim([-48, 48])
    plt.legend()
    plt.title(title)

    # Reset rcParams after plotting to avoid affecting other plots
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

# Example usage with custom font size
# bland_altman_plot_multiple(data1, data2_dict, font_size=18)
