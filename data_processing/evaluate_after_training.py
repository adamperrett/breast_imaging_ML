print("Starting imports")
import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import sys
import os
import re
import pydicom
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import filters
import torchvision.transforms as T
import optuna
import sqlite3
import os
from math import log10, floor

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
parent_dir = os.path.dirname(current_path)
sys.path.insert(0, parent_dir)
from training.training_config import *
from models.architectures import *
from models.dataloading import *
from data_processing.data_analysis import *
# from data_processing.dcm_processing import *
from data_processing.plotting import *

criterion = nn.MSELoss(reduction='none')

testing_data = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/priors_pvas_vbd_processed_per_im_base.pth'

model_location = 'C:/Users/adam_/OneDrive/Documents/Research/Breast cancer/results/full volpara/models'
# model_name = 'l_big_filterCC_lr0.00067x15_pvas_p1r0_drop0.0329_adam_t0_w0'
model_name = 'l_volpara_weight_lr1.99e-05x26_resnetrans_p1r1_d0.254_adam_t0_wl0_ws0'
test_name = model_name
split = 0
pretrain = 1
replicate = 1
dropout = 0.0459
batch_size = 32

print('Loading model')
model = ResNetTransformer(pretrain=False, replicate=replicate, dropout=dropout, split=split).to('cuda')
# model = Pvas_Model(pretrain=False, replicate=replicate, dropout=dropout).to('cuda')
model.load_state_dict(torch.load(os.path.join(current_path, 'best_models', model_name)))

print('Making dataloader')
training_mean, training_std = 6.448529270685709, 4.314404263091535
data_loader = return_dataloaders(testing_data, transformed=0,
                                 weighted_loss=0, weighted_sampling=0,
                                 batch_size=batch_size, only_testing=True)

print("Evaluating model")
loss, labels, preds, r2, file_names = evaluate_model(model, data_loader, criterion,
                                                     inverse_standardize_targets, training_mean, training_std,
                                                     return_names=True, split_CC_and_MLO=split)

print("loss:", loss)
print("r2:", r2)

print("Splitting into different views and files")

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
csv_name = 'PROCAS_Volpara_dirty.csv'
procas_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

data_dict = {'ASSURE_PROCESSED_ANON_ID': [], 'view': [], 'value': []}

# Extract ID and view from file names
for file_name, value in zip(file_names, preds):
    # Adjusted regex to match 3 or 4 capital letters for the view code
    match = re.search(r'-([0-9]{5})-([R|L][C|M][C|L][O]?)', file_name)
    if match:
        id_ = match.group(1)
        view_code = match.group(2)
        data_dict['ASSURE_PROCESSED_ANON_ID'].append(int(id_))
        data_dict['view'].append(view_code)
        data_dict['value'].append(value)

# Convert the dictionary to a DataFrame
extracted_data = pd.DataFrame(data_dict)

# Define view to column mapping, appending "_test" to each column name
view_column_mapping = {
    'LCC': f'vbd_L_CC_{test_name}',
    'RCC': f'vbd_R_CC_{test_name}',
    'LMLO': f'vbd_L_MLO_{test_name}',
    'RMLO': f'vbd_R_MLO_{test_name}'
}

# Merge the extracted data into the original DataFrame
for view_code, column_name in view_column_mapping.items():
    temp_df = extracted_data[extracted_data['view'] == view_code].copy()
    temp_df.rename(columns={'value': column_name}, inplace=True)
    temp_df.drop('view', axis=1, inplace=True)
    procas_data = pd.merge(procas_data, temp_df, on='ASSURE_PROCESSED_ANON_ID', how='left')

# Calculate the VBD as the average of the four new test views, adding "_test" to the "VBD" column name
procas_data[f'VBD_{test_name}'] = procas_data[[view_column_mapping['LCC'], view_column_mapping['RCC'], view_column_mapping['LMLO'], view_column_mapping['RMLO']]].mean(axis=1)

# Save the updated dataframe to a new CSV file
output_csv_name = 'volpara_priors_testing_weight.csv'  # Update this filename as needed
procas_data.to_csv(os.path.join(csv_directory, output_csv_name), index=False)

print(f"Updated data saved to {os.path.join(csv_directory, output_csv_name)}")

print("Done")
