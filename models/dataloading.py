import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import os
import re
import pydicom
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from tqdm import tqdm
from skimage import filters, exposure
from copy import deepcopy
import torchvision.transforms as T
from pathlib import Path
import os
from torch.utils.data import Dataset, Subset
from skimage.transform import resize
import torch
import re
import time
import pydicom
from skimage import filters
import gc
import psutil
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
parent_dir = os.path.dirname(current_path)
sys.path.insert(0, parent_dir)
from training.training_config import *
from data_processing.data_analysis import *


class MammogramDataset(Dataset):
    def __init__(self, dataset_path, transform=None, n=0, weights=None, rand_select=True, no_process=False):
        self.dataset = torch.load(dataset_path)
        self.transform = transform
        self.n = n
        self.weights = weights
        self.rand_select = rand_select
        self.no_process = no_process

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.no_process:
            return self.dataset[idx]
        image, label, directory, views = self.dataset[idx]

        if self.n:
            if len(image) < self.n:
                for i in range(self.n - len(image)):
                    image = torch.vstack([image, torch.zeros_like(image[0]).unsqueeze(0)])
            if self.rand_select:
                sample = random.sample(range(len(image)), self.n)
            else:
                sample = range(self.n)
            if self.transform:
                transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image[sample]]
                image = transformed_image
            else:
                image = image[sample]
        else:
            if self.transform:
                transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image]
                image = transformed_image

        return image, label, directory, views

def custom_collate(batch):
    # Separate images and labels
    images, labels, weights = zip(*batch)

    # Determine the max combined width
    max_width = max([sum(img.size(-1) for img in img_list) for img_list in images])

    # Stack images horizontally with padding
    stacked_images = []
    for img_list in images:
        combined_width = sum(img.size(-1) for img in img_list)
        padding_size = max_width - combined_width
        combined_img = torch.cat(tuple(img_list), dim=-1)  # Use tuple() here
        if padding_size > 0:
            # Pad on the right
            combined_img = pad(combined_img, (0, 0, padding_size, 0))
        stacked_images.append(combined_img)

    # Stack the processed images into a batch
    images_tensor = torch.stack(stacked_images)

    # Convert the list of regression targets to a tensor and standardize
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Change the dtype if needed
    labels_tensor = standardize_targets(labels_tensor, mean, std)

    # if weights[0]:
    return images_tensor, labels_tensor, torch.tensor(weights, dtype=torch.float32)


print("Reading data")

process_type = 'log'#, 'histo', 'clahe'

csf = True
if csf:
    csv_directory = '/mnt/bmh01-rds/assure/csv_dir/'
    csv_name = 'pvas_data_sheet.csv'
    save_dir = '/mnt/bmh01-rds/assure/processed_data/'
    save_name = 'procas_all'
else:
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
    csv_name = 'priors_per_image_reader_and_MAI.csv'
    save_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'
    save_name = 'priors'

if by_patient:
    save_name += '_patients'
if pvas_loader:
    save_name += '_pvas'

procas_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

if raw:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_RAW'
    else:
        image_directory = 'D:/priors_data/raw'
    procas_ids = procas_data['ASSURE_RAW_ID']
    save_name += '_raw'
else:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_PROCESSED'
    else:
        image_directory = 'D:/priors_data/processed'
    procas_ids = procas_data['ASSURE_PROCESSED_ANON_ID']
    save_name += '_processed'

if not raw or pvas_loader:
    process_types = 'base'
save_name += '_'+process_type


# Load dataset from saved path
print("Creating Dataset")
dataset = MammogramDataset(processed_dataset_path,
                           n=n_images)

# Splitting the dataset
train_ratio, val_ratio = 0.7, 0.2
num_train = int(train_ratio * len(dataset))
num_val = int(val_ratio * len(dataset))
num_test = len(dataset) - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

if transformed:
    # Define your augmentations
    data_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        transforms.RandomCrop(size=(10 * 64, 8 * 64), padding=4),
        # Add any other desired transforms here
    ])
else:
    data_transforms = None

# Compute weights for the training set
if weighted:
    targets = [label for _, label in train_dataset.dataset.dataset]
    sample_weights = compute_sample_weights(targets)
else:
    sample_weights = None

# Applying the transform only to the training dataset
train_dataset.dataset = MammogramDataset(processed_dataset_path,
                                         transform=data_transforms,
                                         weights=sample_weights,
                                         n=parallel_images)

mean, std = compute_target_statistics(train_dataset)

# from torch.utils.data import WeightedRandomSampler
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))

# Use this sampler in your DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate)

# Create DataLoaders
print("Creating DataLoaders")
if by_patient:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)