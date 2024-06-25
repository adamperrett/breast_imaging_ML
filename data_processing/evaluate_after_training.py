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

threshold = '40'

testing_data = '../processed_data/priors_pvas_vas_raw_base_CC.pth'

model_location = '../models/'
# model_name = 'l_big_filterCC_lr0.00067x15_pvas_p1r0_drop0.0329_adam_t0_w0'
# model_name = 'l_volpara_weight_lr1.99e-05x26_resnetrans_p1r1_d0.254_adam_t0_wl0_ws0'
# model_name = 'l_ViTCC_lr1.64e-05x20_vit_p1r0_d0.562_adam_t0_wl0_ws0'
model_name = 'l_thresholdCC_lr3.01e-05x9_pvas_p1r0_d0.0588_adam_t0_wl0_ws0_th' + threshold 
test_name = model_name
split = 1
pretrain = 1
replicate = 0
dropout = 0.575
batch_size = 1 
isRaw = True

if isRaw:
    matchColumn = 'ASSURE_RAW_ID'
else:
    matchColumn = 'ASSURE_PROCESSED_ANON_ID'
print('Loading model')
# model = ResNetTransformer(pretrain=False, replicate=replicate, dropout=dropout, split=split).to('cuda')
model = Pvas_Model(pretrain=False, replicate=replicate, dropout=dropout, split=True).to('cpu')
# model = ViT_Model().to('cpu')
state_dict = torch.load(os.path.join(model_location, model_name), map_location=torch.device('cpu')) 
#state_dict['extractor.conv1.1.weight'] = state_dict.pop('extractor.conv1.weight')

model.load_state_dict(state_dict)

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

print(len(labels))
print(len(preds))

print("Splitting into different views and files")

csv_directory = '/mnt/bmh01-rds/assure/csv_dir/'
csv_name = 'PROCAS_matched_priors_v2.csv'
procas_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

data_dict = {matchColumn: [], 'view': [], 'value': []}

# Extract ID and view from file names
for file_name, value in zip(file_names, preds):
    # Adjusted regex to match 3 or 4 capital letters for the view code
    match = re.search(r'-([0-9]{5})-([R|L][C|M][C|L][O]?)', file_name)
    if match:
        id_ = match.group(1)
        view_code = match.group(2)
        data_dict[matchColumn].append(int(id_))
        data_dict['view'].append(view_code)
        data_dict['value'].append(value)

# Convert the dictionary to a DataFrame
extracted_data = pd.DataFrame(data_dict)

print("extracted_data")
print(extracted_data)

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
    procas_data = pd.merge(procas_data, temp_df, on=matchColumn, how='left')

# Calculate the VBD as the average of the four new test views, adding "_test" to the "VBD" column name
procas_data[f'VBD_{test_name}'] = procas_data[[view_column_mapping['LCC'], view_column_mapping['RCC'], view_column_mapping['LMLO'], view_column_mapping['RMLO']]].mean(axis=1)
print(len(procas_data))
# Save the updated dataframe to a new CSV file
output_csv_name = 'Thresh' + threshold + '_priors_testing.csv'  # Update this filename as needed
procas_data.to_csv(os.path.join('./', output_csv_name), index=False)

print(f"Updated data saved to {os.path.join('./', output_csv_name)}")

print("Done")
