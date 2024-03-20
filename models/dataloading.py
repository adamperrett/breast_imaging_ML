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

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'



class MammogramDataset(Dataset):
    def __init__(self, dataset, transform=None, n=0, weights=None, rand_select=True, no_process=False):
        self.dataset = dataset #torch.load(dataset_path)
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

        if self.transform:
            transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image]
            image = transformed_image

        if self.weights is not None:
            return image, label, self.weights[idx], directory, views
        else:
            return image, label, torch.ones_like(self.weights[idx]), directory, views

def custom_collate(batch):
    # Separate images and labels
    images, labels, weights, dir, view = zip(*batch)

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

mean = 0
std = 0

def split_by_patient(dataset_path, train_ratio, val_ratio, seed_value=0):
    dataset = torch.load(dataset_path)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Group entries by the unique key
    groups = {}
    for entry in dataset:
        key = entry[2]  # The third entry
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # Shuffle the list of keys for randomness
    keys = list(groups.keys())
    random.shuffle(keys)

    # Calculate the split sizes based on the number of unique keys
    train_end = int(len(keys) * train_ratio)
    val_end = train_end + int(len(keys) * val_ratio)

    # Split the keys into train, validation, and test
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    # Select data from the original dataset based on the split keys
    train_data = [item for k in train_keys for item in groups[k]]
    val_data = [item for k in val_keys for item in groups[k]]
    test_data = [item for k in test_keys for item in groups[k]]

    return train_data, val_data, test_data

def return_dataloaders(processed_dataset_path, transformed, weighted, seed_value=0):
    global mean, std

    # Splitting the dataset
    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    train_data, val_data, test_data = split_by_patient(processed_dataset_path, train_ratio, val_ratio, seed_value)

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Compute weights for the training set
    if weighted:
        targets = [label for _, label, _, _ in train_data]
        sample_weights = compute_sample_weights(targets)
    else:
        sample_weights = None

    # Load dataset from saved path
    print("Creating Dataset")
    train_dataset = MammogramDataset(train_data, transform=data_transforms, weights=sample_weights)
    val_dataset = MammogramDataset(val_data)
    test_dataset = MammogramDataset(test_data)

    mean, std = compute_target_statistics(train_dataset)

    # from torch.utils.data import WeightedRandomSampler
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))

    # Use this sampler in your DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate)

    # Create DataLoaders
    print("Creating DataLoaders")
    if by_patient:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                  generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader