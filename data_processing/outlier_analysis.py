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


seed_value = 272727
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

print("Reading data")

raw = False  # Raw or processed data
creating_pvas_loader = True  # if true process types makes no difference
by_patient = False  # DEPRICATED: Put all patient images into a single data instance
split_CC_and_MLO = False  # Create a separate dataset for CC and MLO or combine it all
average_score = False  # Do you want per image scores or average over all views
clean_with_pvas = False  # Will keep only patients in the clean pvas datasheet
remove_priors = True  # Will the dataset filter out priors
use_priors = False
if use_priors:
    remove_priors = False

vas_or_vbd = 'vas'

priors_csv = 'PROCAS_matched_priors_v2.csv'

csf = False
if csf:
    csv_directory = '/mnt/bmh01-rds/assure/csv_dir/'
    save_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data'
    if use_priors:
        save_name = 'priors'
    else:
        save_name = 'procas'
else:
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
    save_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'
    save_name = 'local'
if vas_or_vbd == 'vas':
    if use_priors:
        csv_name = priors_csv
    else:
        csv_name = 'pvas_data_sheet.csv'
else:
    csv_name = 'PROCAS_Volpara_dirty.csv'
priors_data = pd.read_csv(os.path.join(csv_directory, priors_csv), sep=',')

if by_patient:
    save_name += '_patients'
if creating_pvas_loader:
    save_name += '_pvas'
save_name += '_'+vas_or_vbd

if use_priors:
    selected_patients = pd.read_csv(os.path.join(csv_directory, priors_csv), sep=',')
else:
    selected_patients = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
procas_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
clean_pvas_data = pd.read_csv(os.path.join(csv_directory, 'pvas_data_sheet.csv'), sep=',')

if raw:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_RAW'
    else:
        image_directory = 'Z:/PROCAS_ALL_RAW'
    image_type_id = 'ASSURE_RAW_ID'
    save_name += '_raw'
else:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_PROCESSED'
    else:
        # image_directory = 'Z:/PROCAS_ALL_PROCESSED'
        image_directory = 'D:/priors_data/raw'
    image_type_id = 'ASSURE_PROCESSED_ANON_ID'
    save_name += '_processed'
procas_ids = procas_data[image_type_id]
selected_ids = selected_patients[image_type_id]
clean_pvas_ids = clean_pvas_data[image_type_id]
priors_ids = priors_data[image_type_id]

def format_id(id):
    # Ensure the id is an integer and format it
    # return "{:05}".format(int(id))
    return id

# Drop NaN values, then apply the formatting function
clean_pvas_ids = set(clean_pvas_ids.dropna().apply(format_id).values)
selected_ids = set(selected_ids.dropna().apply(format_id).values)
priors_ids = set(priors_ids.dropna().apply(format_id).values)

regression_target_data = {}
if vas_or_vbd == 'vas':
    if average_score or use_priors:
        regression_target_data['LMLO'] = procas_data['VASCombinedAvDensity']
        regression_target_data['LCC'] = procas_data['VASCombinedAvDensity']
        regression_target_data['RMLO'] = procas_data['VASCombinedAvDensity']
        regression_target_data['RCC'] = procas_data['VASCombinedAvDensity']
    else:
        regression_target_data['LMLO'] = procas_data['LMLO']
        regression_target_data['LCC'] = procas_data['LCC']
        regression_target_data['RMLO'] = procas_data['RMLO']
        regression_target_data['RCC'] = procas_data['RCC']
        regression_target_data['LMLO-1'] = procas_data['LMLO-1']
        regression_target_data['LCC-1'] = procas_data['LCC-1']
        regression_target_data['RMLO-1'] = procas_data['RMLO-1']
        regression_target_data['RCC-1'] = procas_data['RCC-1']
        regression_target_data['LMLO-2'] = procas_data['LMLO-2']
        regression_target_data['LCC-2'] = procas_data['LCC-2']
        regression_target_data['RMLO-2'] = procas_data['RMLO-2']
        regression_target_data['RCC-2'] = procas_data['RCC-2']
else:
    if average_score:
        save_name += '_average'
        regression_target_data['LMLO'] = procas_data['VBD']
        regression_target_data['LCC'] = procas_data['VBD']
        regression_target_data['RMLO'] = procas_data['VBD']
        regression_target_data['RCC'] = procas_data['VBD']
    else:
        save_name += '_per_im'
        regression_target_data['LMLO'] = procas_data['vbd_L_MLO']
        regression_target_data['LCC'] = procas_data['vbd_L_CC']
        regression_target_data['RMLO'] = procas_data['vbd_R_MLO']
        regression_target_data['RCC'] = procas_data['vbd_R_CC']

# save patient_ids which have a regression target
id_target_dict = {}
for image_type in regression_target_data:
    id_target_dict[image_type] = {}
    for id, target in zip(procas_ids, regression_target_data[image_type]):
        if not np.isnan(id) and not np.isnan(target) and target >= 0:
            if format_id(id) in selected_ids:
                id_target_dict[image_type][format_id(id)] = target

# Find common IDs across all image types
common_ids = set(id_target_dict[next(iter(id_target_dict))])  # Initialize with the first image type's IDs
for image_type, ids in id_target_dict.items():
    common_ids &= set(ids.keys())  # Intersect with the IDs of the current image type

# Additional filtering based on 'filter_priors'
if clean_with_pvas:
    # If filtering exclusion, keep shared keys in 'clean_pvas_data' from 'common_ids'
    common_ids &= clean_pvas_ids
# Additional filtering based on priors
if remove_priors:
    common_ids -= priors_ids

# Apply the final set of 'common_ids' to filter 'id_target_dict'
for image_type in id_target_dict:
    id_target_dict[image_type] = {id: target for id, target in id_target_dict[image_type].items() if id in common_ids}

def calculate_the_median_differences(values):
    diffs = {}
    sq_diffs = {}
    for view_1 in values:
        diff = 0
        sq_diff = 0
        for view_2 in values:
            if view_1 != view_2:
                diff += abs(values[view_1] - values[view_2])
                sq_diff += np.square(values[view_1] - values[view_2])
        diffs[view_1] = diff
        sq_diffs[view_1] = sq_diff
    average_diff = np.mean(list(diffs.values()))
    median_diff = (np.sum(list(diffs.values()))
                   - np.min(list(diffs.values()))
                   - np.max(list(diffs.values()))) / 2
    median_dist = {key: abs(value - median_diff) for key, value in diffs.items()}
    average_sq_diff = np.mean(list(sq_diffs.values()))
    median_sq_diff = (np.sum(list(sq_diffs.values()))
                      - np.min(list(sq_diffs.values()))
                      - np.max(list(sq_diffs.values()))) / 2
    median_sq_dist = {key: abs(value - median_sq_diff) for key, value in sq_diffs.items()}
    return diffs, average_diff, median_diff, median_dist, sq_diffs, average_sq_diff, median_sq_diff, median_sq_dist

def filter_with_comparative_difference(patient, target_dict, thresholds, view=''):
    view_type = ['MLO-1', 'CC-1', 'MLO-2', 'CC-2']
    values = {view+v: target_dict[view+v][patient] for v in view_type}
    diffs = {}
    sq_diffs = {}
    # outlier = {th: {} for th in thresholds}
    diffs, average_diff, median_diff, median_dist, sq_diffs, average_sq_diff, median_sq_diff, median_sq_dist = \
        calculate_the_median_differences(values)
    _, _, _, m_median_dist, _, _, _, m_median_sq_dist = \
        calculate_the_median_differences(median_dist)
    # for d in diffs:
    #     for th in thresholds:
    #         if abs(diffs[d] - average_diff) > th:
    #             outlier[th][d] = 1
    #         else:
    #             outlier[th][d] = 0
    return diffs, average_diff, median_diff, median_dist, m_median_dist, sq_diffs, average_sq_diff, median_sq_diff, median_sq_dist

def volpara_metrics(four_values):
    dic_values = list(four_values.values())
    summed = np.sum(dic_values)
    minned = np.min(dic_values)
    maxed = np.max(dic_values)
    median = (summed - minned - maxed) / 2
    if maxed == 0:
        threshold = 1
    else:
        threshold = np.sqrt(minned) / np.sqrt(maxed)
    return median, threshold, maxed

def filter_with_sum_of_squared_diff(patient, target_dict, thresholds, view=''):
    view_type = ['MLO-1', 'CC-1', 'MLO-2', 'CC-2']
    values = {view+v: target_dict[view+v][patient] for v in view_type}
    diffs = {v: 0 for v in values}
    # outlier = {th: {} for th in thresholds}
    for view_1 in values:
        done_views = []
        for view_a in values:
            for view_b in values:
                if view_1 != view_a and view_1 != view_b and view_b not in done_views:
                    diffs[view_1] += np.square(values[view_a] - values[view_b]) / 3
            done_views.append(view_a)
    median, threshold, max_ssqd = volpara_metrics(diffs)
    # outlier_by_th =
    return diffs, median, threshold, max_ssqd

views = ['LMLO', 'RMLO', 'LCC', 'RCC']
filter_thresholds = [i+1 for i in range(0, 30, 2)]
# filtered_by_comparative_diff = {ft: id_target_dict for ft in filter_thresholds}
# outliers_by_comparative_diff = {ft: [] for ft in filter_thresholds}
patient_meta_data = {}
for patient in tqdm(id_target_dict['LCC']):
    meta_data = {}
    for view_type in ['L', 'R']:
        abs_diffs, ave_diff, median_diff, median_dist, m_median_dist, \
        sq_diffs, ave_sq_diff, median_sq_diff, median_sq_dist = \
            filter_with_comparative_difference(
                patient,
                id_target_dict,
                filter_thresholds,
                view_type)
        ssqd, median_ssqd, threshold, max_ssqd = \
            filter_with_sum_of_squared_diff(
                patient,
                id_target_dict,
                filter_thresholds,
                view_type)
        meta_data[view_type+'_ave_diff'] = ave_diff
        meta_data[view_type+'_median_diff'] = median_diff
        meta_data[view_type+'_ave_sq_diff'] = ave_sq_diff
        meta_data[view_type+'_median_sq_diff'] = median_sq_diff
        for view in abs_diffs:
            meta_data[view+'_abs_diff'] = abs_diffs[view]
        for view in median_dist:
            meta_data[view+'_median_dist'] = median_dist[view]
        for view in m_median_dist:
            meta_data[view+'_m_median_dist'] = m_median_dist[view]
        for view in sq_diffs:
            meta_data[view+'_abs_sq_diff'] = sq_diffs[view]
        for view in median_sq_dist:
            meta_data[view+'_median_sq_dist'] = median_sq_dist[view]
        meta_data[view_type+'_median_ssqd'] = median_ssqd
        meta_data[view_type+'_threshold'] = threshold
        meta_data[view_type+'_max_ssqd'] = max_ssqd
        for view in ssqd:
            meta_data[view+'_ssqd'] = ssqd[view]

    patient_meta_data[patient] = meta_data

df_meta_data = pd.DataFrame.from_dict(patient_meta_data, orient='index')

# Reset index to have patient ID as a column for merging
df_meta_data.reset_index(inplace=True)

# Rename the index column to match the PROCESSED_ID column in the original CSV
df_meta_data.rename(columns={'index': 'ASSURE_PROCESSED_ANON_ID'}, inplace=True)

merged_df = pd.merge(df_meta_data, procas_data, how='left', on='ASSURE_PROCESSED_ANON_ID')

merged_df.to_csv('../csv_data/outlier_processing.csv', index=False)

print("Done!")