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
from skimage import measure, filters
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

raw = True  # Raw or processed data
creating_pvas_loader = True  # if true process types makes no difference
by_patient = False  # DEPRICATED: Put all patient images into a single data instance
split_CC_and_MLO = False  # Create a separate dataset for CC and MLO or combine it all
average_score = False  # Do you want per image scores or average over all views
clean_with_pvas = False  # Will keep only patients in the clean pvas datasheet
remove_priors = True  # Will the dataset filter out priors
use_priors = True
making_medici = True
if use_priors:
    remove_priors = False

vas_or_vbd = 'vas'

process_types = ['log']#, 'histo', 'clahe']  # only relevant to raw data
priors_csv = 'PROCAS_matched_priors_v2.csv'

csf = True
if csf:
    csv_directory = '/mnt/bmh01-rds/assure/csv_dir/'
    save_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data'
else:
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
    save_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'
save_name = 'medici_preprocessed_data.pth'
csv_name = '_vendors_grouped_Reader_1704subjects.csv'

csv_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

if csf:
    image_directory = '/mnt/bmh01-rds/assure/MEDICI/MEDICI_subset'
else:
    image_directory = 'W:/Reader_study_readings_07_08_2024/dataCopy/MEDICI_subset'

def preprocess_image_medici(image_path, side, format, view):
    """
    Load and preprocess the given image.
    """
    # Read the DICOM file
    current_mammogram = pydicom.read_file(os.path.join(image_directory, image_path))
    ## fetch the pixel array, and Manufacturer
    mammographic_image = current_mammogram.pixel_array
    Manufacturer = current_mammogram.Manufacturer
    # check_image(mammographic_image, '1-Image')

    ## Some mammograms need to be inverted
    if Manufacturer == 'Sectra Imtec AB' or Manufacturer == 'FUJIFILM Corporation' or Manufacturer == 'Philips Digital Mammography Sweden AB' or Manufacturer == 'Philips Medical Systems':
        mammographic_image = np.amax(mammographic_image) - mammographic_image
        # check_image(mammographic_image, '1_1-Image inverted')
    ## Right side mammograms are flipped
    if side == 'R':
        mammographic_image = np.fliplr(mammographic_image)
    ## For PROC GE, apply otsu's
    if format == 'PRO' and "GE" in Manufacturer:
        cut_off = filters.threshold_otsu(mammographic_image)
        mammographic_image = np.clip(mammographic_image, cut_off, np.amax(mammographic_image))
        # check_image(mammographic_image[:, :], '3-GE image After applying Otsu')
    elif format == 'PRO' and "GE" not in Manufacturer:
        thresh = filters.threshold_triangle(mammographic_image)
        blobs = mammographic_image > thresh
        all_labels = measure.label(blobs)
        blobs_labels = measure.label(blobs, background=0)
        unique_values, counts = np.unique(blobs_labels, return_counts=True)
        max_index = np.argmax(counts[1:])
        max_index = max_index + 1
        max_blob = blobs_labels
        max_blob[max_blob != max_index] = 0
        max_blob[max_blob == max_index] = 1
        mammographic_image = mammographic_image * max_blob
        # check_image(mammographic_image[:, :], '3-'+str(Manufacturer)+' image After applying threshold_triangle and tag removal')
    # process to pixel size=0.0941
    height = current_mammogram.Rows
    width = current_mammogram.Columns
    pixel_size = current_mammogram.ImagerPixelSpacing
    MF_cal = 1
    # DO NOT CHANGE THIS!!!!
    target_pixel_size = 0.0941  # All models have been trained on images with this pixel size
    new_height = int(np.ceil(mammographic_image.shape[0] * pixel_size[0] / (target_pixel_size * MF_cal)))
    new_width = int(np.ceil(mammographic_image.shape[1] * pixel_size[1] / (target_pixel_size * MF_cal)))
    max_intensity = np.amax(mammographic_image)
    mammographic_image = resize(mammographic_image, (new_height, new_width))
    # Rescale intensity values to their original range
    image = mammographic_image * max_intensity / np.amax(mammographic_image)
    image = (image - image.min()) / (image.max() - image.min())
    # print(image.min())
    # check_image(image[:, :], '4-After pixel resizing')
    ## Pad images to the same size before resizing
    padded_image = np.zeros((np.amax([2995, image.shape[0]]), np.amax([2394, image.shape[1]])))
    padded_image[0:image.shape[0], 0:image.shape[1]] = image[:, :]
    image = padded_image[0:2995, 0:2394]
    # check_image(image[:, :], '5-After padding')
    ## Resize to this precise dimension
    image = resize(image, (640, 512))
    # check_image(image[:, :], '6-After resizing to 640 x 512')

    ## Max min normalise
    image = image / np.amax(image)
    # check_image(image[:, :], '7-After Min max normalisation')

    ## Replicate accross the channels
    image = np.stack((image, image, image), 0)

    if ('GE' in Manufacturer or Manufacturer == 'KODAK' or Manufacturer == 'LORAD'):
        ## Normalise to 0 mean 1 s.d. These are PROCAS means (training)
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.147, 0.147, 0.147), (0.261, 0.261, 0.261))
        elif format == 'RAW':
            norm = T.Normalize((0.183, 0.183, 0.183), (0.331, 0.331, 0.331))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.185, 0.185, 0.185), (0.275, 0.275, 0.275))
            elif format == 'RAW':
                norm = T.Normalize((0.216, 0.216, 0.216), (0.331, 0.331, 0.331))
    if (Manufacturer == 'HOLOGIC, Inc.'):
        ## Normalise to 0 mean 1 s.d. These are calculated from 1000 MEDICI mamos.
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.073, 0.073, 0.073), (0.147, 0.147, 0.147))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.108, 0.108, 0.108), (0.182, 0.182, 0.182))
    if (Manufacturer == 'SIEMENS'):
        ## Normalise to 0 mean 1 s.d. These are calculated from 1000 MEDICI mamos.
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.101, 0.101, 0.101), (0.186, 0.186, 0.186))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.134, 0.134, 0.134), (0.205, 0.205, 0.205))
    if (Manufacturer == 'FUJIFILM Corporation'):
        ## Normalise to 0 mean 1 s.d. These are calculated from all MEDICI mamos.
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.128, 0.128, 0.128), (0.227, 0.227, 0.227))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.148, 0.148, 0.148), (0.23, 0.23, 0.23))
    if (Manufacturer == 'IMS s.r.l.'):
        ## Normalise to 0 mean 1 s.d. These are calculated from all MEDICI mamos.
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.199, 0.199, 0.199), (0.339, 0.339, 0.339))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.275, 0.275, 0.275), (0.366, 0.366, 0.366))

    if (
            Manufacturer == 'Philips Digital Mammography Sweden AB' or Manufacturer == 'Philips Medical Systems' or Manufacturer == 'Sectra Imtec AB'):
        ## Normalise to 0 mean 1 s.d. These are calculated from all MEDICI mamos.
        if view == 'CC':
            if format == 'PRO':
                norm = T.Normalize((0.146, 0.146, 0.146), (0.265, 0.265, 0.265))
        elif view == 'MLO':
            if format == 'PRO':
                norm = T.Normalize((0.183, 0.183, 0.183), (0.280, 0.280, 0.280))

    image = torch.as_tensor(image.copy())
    image = norm(image)
    # check_image(image[0,:,:].numpy(), '7-After vendor-specific normalization')
    return image

if __name__ == "__main__":
    patient_data = {}
    for index, row in tqdm(csv_data.iterrows(), total=len(csv_data), desc="Processing rows"):
        patient = row.Case
        if patient not in patient_data:
            patient_data[patient] = {}
        side = row.Side
        manufacturer = row.Manufacturer
        score = row.Score
        time_point = row.TimePoint
        format = 'PRO'
        cc_path = row.CCpath
        view = 'CC'
        cc_image = preprocess_image_medici(cc_path, side, format, view)
        mlo_path = row.MLOpath
        view = 'MLO'
        mlo_image = preprocess_image_medici(mlo_path, side, format, view)
        patient_data[patient][time_point] = {
            'mlo': mlo_image,
            'cc': cc_image,
            'score': score,
            'manufacturer': manufacturer
        }
    save_location_and_name = os.path.join(save_dir, save_name)
    print("Saving to", save_location_and_name, "with", len(patient_data), 'women')
    torch.save(patient_data, save_location_and_name)

    print("Done")

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import MultilabelStratifiedShuffleSplit

# Sample DataFrame with multiple targets (replace this with your actual data)
# Assuming `X` is your features DataFrame and `Y` is a DataFrame of multiple target labels
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10]
})
targets = pd.DataFrame({
    'target1': [1, 0, 1, 0, 1],
    'target2': [0, 1, 0, 1, 1]
})

# Initialize MultilabelStratifiedShuffleSplit
splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Perform the split
for train_index, test_index in splitter.split(data, targets):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

# Display results
print("Training features:\n", X_train)
print("Training targets:\n", y_train)
print("Testing features:\n", X_test)
print("Testing targets:\n", y_test)
'''