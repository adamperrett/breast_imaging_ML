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
from scipy.stats import pearsonr

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
# testing_data = 'raw_mosaic_dataset_log'
testing_data = 'priors_pvas_vas_raw_base'
# model_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/models/best_models'
data_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/models/mosaic_models'
plot_save_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/data_processing/plots'
if 'priors' in testing_data:
    plot_save_location += '/priors_test'
else:
    plot_save_location += '/mosaic_test'
save_not_plot = True
# model_name = 'l_big_filter_lr1.9e-05x8_resnetrans_p1r0_drop0.0459_adam_t0_w0'
# model_name = 'l_mosaic_n4_testing_lr0.000169x17_18attention_p1r0_d0.403_adam_t0_wl0_ws0'
# model_name = 'l_mosaic_n1_pvas_testing_lr6.33e-05x11_34mean_p1r0_d0.59_adam_t0_wl0_ws0'
model_names = [
    # 'l_mosaic_n1_pvas_testing_lr6.33e-05x11_34mean_p1r0_d0.59_adam_t0_wl0_ws0',
    'l_mosaic_n4_testing_lr0.000169x17_18attention_p1r0_d0.403_adam_t0_wl0_ws0',
    # 'l_mosaic_combined_pvas_testing_lr9.22e-05x9_18mean_p1r0_d0.00526_adam_t0_wl0_ws0',
    'l_mosaic_combined_pvas_testing_lr2.44e-05x17_34attention_p1r0_d0.0924_adam_t0_wl0_ws0',
    # 'l_mosaic_combined_pvas_testing_lr3.93e-05x8_18mean_p1r0_d0.163_adam_t0_wl0_ws0'
    'l_non_mosaic_mil_testing_lr0.000338x18_34attention_p1r0_d0.283_adam_t0_wl0_ws0'
]
# base_model_name = 'PVAS mosaic model'
base_model_names = [
    # 'PVAS mosaic',
    'MIL unclean',
    # 'MIL combined18a',
    'MIL combined',
    # 'MIL combined18b',
    'MIL clean',
    'pVAS clean'
]
all_model_string = base_model_names[0]
for bmn in base_model_names[1:]:
    all_model_string += f' vs. {bmn}'
# arch = 'pvas'
archs = [
    # 'pvas',
    'mil', 'mil', 'mil']
resnet_sizes = [
    # 34,
    18, 34, 34]
pooling_type = 'attention'
# test_name = model_name
split = 0
pretrain = 1
replicate = 0
dropout = 0.159
batch_size = 8

print('Loading model')
models = []
for model_name, arch, resnet_size in zip(model_names, archs, resnet_sizes):
    if arch == 'resnetrans':
        model = ResNetTransformer(pretrain=False, replicate=replicate, dropout=dropout, split=split).to('cuda')
    if arch == 'mil':
        model = Mosaic_MIL_Model(pretrain=False, replicate=replicate, dropout=dropout, split=split,
                                 resnet_size=resnet_size, pooling_type=pooling_type).to('cuda')
    elif arch == 'pvas':
        model = Mosaic_PVAS_Model(pretrain=False, replicate=replicate, dropout=dropout, split=split,
                                  resnet_size=resnet_size).to('cuda')
    model.load_state_dict(torch.load(os.path.join(data_location, model_name)))
    model.eval()
    models.append(model)
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
running_loss = {bmn: 0.0 for bmn in base_model_names}
all_targets = []
all_predictions = {bmn: [] for bmn in base_model_names}
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
        for model, bmn in zip(models, base_model_names):
            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo.cuda()).to('cpu')
            test_outputs_original_scale = outputs.squeeze(1)  # inverse_standardize_targets(outputs.squeeze(1), mean, std)
            test_targets_original_scale = targets.float()  # inverse_standardize_targets(targets.float(), mean, std)
            loss = criterion(test_outputs_original_scale, test_targets_original_scale).mean()
            running_loss[bmn] += loss.item() * inputs.size(0)
            all_predictions[bmn].extend(test_outputs_original_scale.cpu().numpy())
        test_pvas_outputs_original_scale = torch.mean(torch.stack(pvas_output))
        pvas_loss = criterion(test_pvas_outputs_original_scale, test_targets_original_scale).mean()
        running_loss['pVAS clean'] += pvas_loss.item() * inputs.size(0)
        all_predictions['pVAS clean'].extend(test_pvas_outputs_original_scale.unsqueeze(0).cpu().numpy())

        all_targets.extend(test_targets_original_scale.cpu().numpy())
        all_names.append(file_names)
        all_dirs.extend(dir)

epoch_loss = {bmn: running_loss[bmn] / len(test_loader.dataset) for bmn in base_model_names}
if torch.sum(torch.isnan(torch.tensor(all_targets))) > 0:
    print("Corrupted targets")
for bmn in base_model_names:
    if torch.sum(torch.isnan(torch.tensor(all_predictions[bmn]))) > 0:
        print("Corrupted predictions")
# error, stderr, conf_int = compute_error_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))
# errors = torch.tensor(all_targets) - torch.tensor(all_predictions)
r2 = {bmn: r2_score(all_targets, all_predictions[bmn]) for bmn in base_model_names}

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
csv_name = 'APEP_sofia.csv'
full_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')


print("loss:", epoch_loss)
print("r2:", r2)

print("Splitting into different views and files")

patient_ids = all_dirs  # list of patient IDs
# errors = torch.tensor(all_targets) - torch.tensor(all_predictions)
targets = torch.tensor(all_targets)
# predictions = torch.tensor(all_predictions)

# Prepare a dataframe for analysis
data = pd.DataFrame({
    'patient_id': [int(ID[-5:]) for ID in patient_ids],
    'targets': targets.numpy(),
})
for bmn in base_model_names:
    data[bmn+' errors'] = (torch.tensor(all_targets) - torch.tensor(all_predictions[bmn])).numpy()
    data[bmn+' predictions'] = (torch.tensor(all_predictions[bmn])).numpy()

# Overall Bland-Altman Plot for predictions and pvas
title = f'Overall Bland-Altman Plot'
bland_altman_plot_multiple(
    title,
    data['targets'],
    {bmn: data[bmn+' predictions'] for bmn in base_model_names}
)
if not save_not_plot:
    plt.show()
else:
    plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
    plt.close()

# Merge with full_data to get BMI_category and BMI for each patient
merged_data = pd.merge(data, full_data, left_on='patient_id', right_on='ASSURE_PROCESSED_ANON_ID')

# Task 1: Bland-Altman plots for BMI categories
bmi_categories = merged_data['BMI_category'].unique()

for category in bmi_categories:
    category_data = merged_data[merged_data['BMI_category'] == category]
    title = f'Bland-Altman Plot for BMI Category={category}'
    bland_altman_plot_multiple(
        title,
        category_data['targets'],
        {bmn: category_data[bmn+' predictions'] for bmn in base_model_names}
    )
    if not save_not_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
        plt.close()

# Task 2: Scatter plot for continuous BMI values
filtered_data = merged_data[merged_data['BMI'] > 0]

plt.figure(figsize=(15, 9))
colors = plt.cm.get_cmap('tab10', len(base_model_names))
for i, bmn in enumerate(base_model_names):
    sns.regplot(x=filtered_data['BMI'], y=filtered_data[bmn+' errors'], ci=None,
                line_kws={"color": colors(i)},
                scatter_kws={"color": colors(i)},
                label=f'{bmn}')

# Calculate and display the Pearson correlation coefficient for both
pearson = {bmn: pearsonr(filtered_data['BMI'], filtered_data[bmn+' errors']) for bmn in base_model_names}

title = f'Scatter Plot of Errors vs. BMI'
added_title = ''
for bmn in base_model_names:
    added_title += f'\n{bmn} Correlation: {pearson[bmn][0]:.2f} (p={pearson[bmn][1]:.2e})'

plt.title(title+added_title)
plt.xlabel('BMI')
plt.ylabel('Prediction Error')
plt.legend()
if not save_not_plot:
    plt.show()
else:
    plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
    plt.close()

# Task 3: Bland-Altman plots for Mamm_Shape categories
mamm_shapes = merged_data['Mamm_Shape'].unique()

for shape in mamm_shapes:
    shape_data = merged_data[merged_data['Mamm_Shape'] == shape]
    title = f'Bland-Altman Plot for Mamm Shape={shape}'
    bland_altman_plot_multiple(
        title,
        shape_data['targets'],
        {bmn: shape_data[bmn + ' predictions'] for bmn in base_model_names}
    )
    if not save_not_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
        plt.close()

cancer_categories = merged_data['Cancer'].unique()
for category in cancer_categories:
    category_data = merged_data[merged_data['Cancer'] == category]
    title = f'Bland-Altman Plot for Cancer={category}'
    bland_altman_plot_multiple(
        title,
        category_data['targets'],
        {bmn: category_data[bmn + ' predictions'] for bmn in base_model_names}
    )
    if not save_not_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
        plt.close()

cancer_categories = merged_data['HRT'].unique()
for category in cancer_categories:
    category_data = merged_data[merged_data['HRT'] == category]
    title = f'Bland-Altman Plot for HRT={category}'
    bland_altman_plot_multiple(
        title,
        category_data['targets'],
        {bmn: category_data[bmn + ' predictions'] for bmn in base_model_names}
    )
    if not save_not_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
        plt.close()

alcohol_categories = merged_data['AlcoholYN'].unique()
for category in alcohol_categories:
    category_data = merged_data[merged_data['AlcoholYN'] == category]
    title = f'Bland-Altman Plot for Alcohol={category}'
    bland_altman_plot_multiple(
        title,
        category_data['targets'],
        {bmn: category_data[bmn + ' predictions'] for bmn in base_model_names}
    )
    if not save_not_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
        plt.close()

filtered_data = merged_data.dropna(subset=['Months from entry to diag'])
plt.figure(figsize=(15, 9))
colors = plt.cm.get_cmap('tab10', len(base_model_names))
for i, bmn in enumerate(base_model_names):
    sns.regplot(x=filtered_data['Months from entry to diag'], y=filtered_data[bmn+' errors'], ci=None,
                line_kws={"color": colors(i)},
                scatter_kws={"color": colors(i)},
                label=f'{bmn}')

# Calculate and display the Pearson correlation coefficient for both
pearson = {bmn: pearsonr(filtered_data['Months from entry to diag'], filtered_data[bmn+' errors']) for bmn in base_model_names}

title = f'Scatter Plot of Errors vs. Months from entry to diagnosis'
added_title = ''
for bmn in base_model_names:
    added_title += f'\n{bmn} Correlation: {pearson[bmn][0]:.2f} (p={pearson[bmn][1]:.2e})'

plt.title(title+added_title)
plt.xlabel('Months from entry to diagnosis')
plt.ylabel('Prediction Error')
plt.legend()
if not save_not_plot:
    plt.show()
else:
    plt.savefig(f"{plot_save_location}/{title}.png", bbox_inches='tight', format='png')
    plt.close()

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
