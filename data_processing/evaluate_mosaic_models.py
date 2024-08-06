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

# testing_data = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/local_pvas_vbd_processed_per_im_base.pth'
testing_data = 'raw_mosaic_dataset_log'
# testing_data = 'priors_pvas_vas_raw_base'
# model_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/models/best_models'
data_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/models/mosaic_models'
# model_name = 'l_big_filter_lr1.9e-05x8_resnetrans_p1r0_drop0.0459_adam_t0_w0'
# model_name = 'l_mosaic_n4_testing_lr0.000169x17_18attention_p1r0_d0.403_adam_t0_wl0_ws0'
model_name = 'l_mosaic_n1_pvas_testing_lr6.33e-05x11_34mean_p1r0_d0.59_adam_t0_wl0_ws0'
arch = 'pvas'
pooling_type = 'attention'
resnet_size = 34
test_name = model_name
split = 0
pretrain = 1
replicate = 0
dropout = 0.159
batch_size = 8

print('Loading model')
if arch == 'resnetrans':
    model = ResNetTransformer(pretrain=False, replicate=replicate, dropout=dropout, split=split).to('cuda')
if arch == 'mil':
    model = Mosaic_MIL_Model(pretrain=False, replicate=replicate, dropout=dropout, split=split,
                             resnet_size=18, pooling_type=pooling_type).to('cuda')
elif arch == 'pvas':
    model = Mosaic_PVAS_Model(pretrain=False, replicate=replicate, dropout=dropout, split=split,
                              resnet_size=resnet_size).to('cuda')
model.load_state_dict(torch.load(os.path.join(data_location, model_name)))
model.eval()
p_model_location = "C:/Users/adam_/PycharmProjects/pVAS/pvas_models/"
p_model_name = 'RAW_model.pth'

cc_model = Pvas_Model(pretrain=True, replicate=True, split=True)
cc_model.cuda()
cc_model = cc_model.double()  ## This is essential
cc_model.load_state_dict(torch.load(os.path.join(p_model_location, 'CC_' + p_model_name), map_location='cuda'))
cc_model.eval()

mlo_model = Pvas_Model(pretrain=True, replicate=True, split=True)
mlo_model.cuda()
mlo_model = mlo_model.double()  ## This is essential
mlo_model.load_state_dict(torch.load(os.path.join(p_model_location, 'MLO_' + p_model_name), map_location='cuda'))
mlo_model.eval()
# model = Pvas_Model(pretrain=False, replicate=replicate, dropout=dropout).to('cuda')

save_path = os.path.join(data_location, testing_data + '_data.pth')
print("Loading data", time.localtime())
data = torch.load(save_path)
print('Making dataloader')
train_data, val_data, test_data = data['train'], data['val'], data['test']
if 'priors' in testing_data:
    test_data.extend(val_data)
    test_data.extend(train_data)
    sorted_by_patient = {}
    for image, vas, r1, r2, patient, file_name in test_data:
        if patient in sorted_by_patient:
            sorted_by_patient[patient][0] = torch.vstack([sorted_by_patient[patient][0], image.unsqueeze(0)])
            sorted_by_patient[patient][-1].append(file_name)
        else:
            sorted_by_patient[patient] = [image.unsqueeze(0), vas, r1, r2, patient, [file_name]]
    test_data = [sorted_by_patient[patient] for patient in sorted_by_patient]
mean, std = data['mean'], data['std']
test_dataset = MosaicEvaluateLoader(test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 generator=torch.Generator(device=device))

print("Evaluating model")
running_loss = 0.0
running_pvas_loss = 0.0
all_targets = []
all_predictions = []
all_pvas = []
all_names = []
all_dirs = []

with torch.no_grad():
    for inputs, targets, w, dir, file_names in tqdm(test_loader):
        nan_mask = torch.isnan(inputs)
        if torch.sum(nan_mask) > 0:
            print("Image is corrupted during evaluation", torch.sum(torch.isnan(inputs), dim=1))
        inputs[nan_mask] = 0
        inputs, targets = inputs.cuda(), targets

        is_it_mlo = torch.zeros([len(file_names[0]), len(file_names), 2]).float()
        pvas_output = []
        for i in range(len(file_names[0])):
            for j in range(len(file_names)):
                pvas_input = torch.stack([inputs[i][j], inputs[i][j], inputs[i][j]]).double().unsqueeze(0)
                if 'MLO' in file_names[j][i]:
                    is_it_mlo[i][j][0] += 1
                    pvas_output.append(
                        mlo_model.forward(pvas_input, None).to('cpu'))
                else:
                    is_it_mlo[i][j][1] += 1
                    pvas_output.append(
                        cc_model.forward(pvas_input, None).to('cpu'))

        outputs = model.forward(inputs.unsqueeze(1), is_it_mlo.cuda()).to('cpu')
        test_outputs_original_scale = outputs.squeeze(1)  # inverse_standardize_targets(outputs.squeeze(1), mean, std)
        test_targets_original_scale = targets.float()  # inverse_standardize_targets(targets.float(), mean, std)
        test_pvas_outputs_original_scale = torch.mean(torch.stack(pvas_output))
        loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
        running_loss += loss.item() * inputs.size(0)
        pvas_loss = criterion(test_pvas_outputs_original_scale, test_targets_original_scale).mean()
        running_pvas_loss += pvas_loss.item() * inputs.size(0)

        all_targets.extend(test_targets_original_scale.cpu().numpy())
        all_predictions.extend(test_outputs_original_scale.cpu().numpy())
        all_pvas.extend(test_pvas_outputs_original_scale.unsqueeze(0).cpu().numpy())
        all_names.append(file_names)
        all_dirs.extend(dir)

epoch_loss = running_loss / len(test_loader.dataset)
epoch_pvas_loss = running_pvas_loss / len(test_loader.dataset)
if torch.sum(torch.isnan(torch.tensor(all_targets))) > 0:
    print("Corrupted targets")
if torch.sum(torch.isnan(torch.tensor(all_predictions))) > 0:
    print("Corrupted predictions")
error, stderr, conf_int = compute_error_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))
r2 = r2_score(all_targets, all_predictions)
r2pvas = r2_score(all_targets, all_pvas)
r2w = r2_score(all_targets, all_predictions, sample_weight=np.array(all_targets) + r2_weighting_offset)

print("loss:", epoch_loss)
print("r2:", r2)
print("pvas loss:", epoch_pvas_loss)
print("pvas r2:", r2pvas)

print("Splitting into different views and files")

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
# csv_name = 'volpara_priors_testing_weight.csv'
# csv_name = 'PROCAS_matched_priors_v2.csv'
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
output_csv_name = 'PROCAS_matched_priors_v2_loss_training.csv'  # Update this filename as needed
procas_data.to_csv(os.path.join(csv_directory, output_csv_name), index=False)

print(f"Updated data saved to {os.path.join(csv_directory, output_csv_name)}")

print("Done")
