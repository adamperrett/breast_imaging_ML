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

raw = True  # Raw or processed data
creating_pvas_loader = True  # if true process types makes no difference

process_types = ['log']#, 'histo', 'clahe']  # only relevant to raw data

csf = True
if csf:
    csv_directory = '/mnt/bmh01-rds/assure/csv_dir/'
    save_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data'
    save_name = 'CRUK'
    # save_name = 'CRUK_full_pp'
else:
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
    save_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'
    save_name = 'CRUK_local'
    # save_name = 'CRUK_local_full_pp'
# =IF(AND(SUM(TD2:TR2)=0,LOWER(GV2)<>"yes"),1,0)
if 'full' in save_name:
    csv_name = 'processed_PROCAS_full_data_with_cancer_data.csv'
else:
    csv_name = 'processed_PROCAS_full_data_only_cancers.csv'
cancer_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

if raw:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_RAW'
    else:
        # image_directory = 'Z:/PROCAS_ALL_RAW'
        image_directory = 'D:/priors_data/raw'
    image_type_id = 'ASSURE_RAW_ID'
    save_name += '_raw'
else:
    if csf:
        image_directory = '/mnt/bmh01-rds/assure/PROCAS_ALL_PROCESSED'
    else:
        # image_directory = 'Z:/PROCAS_ALL_PROCESSED'
        image_directory = 'D:/priors_data/processed'
    image_type_id = 'ASSURE_PROCESSED_ANON_ID'
    save_name += '_processed'

def format_id(id):
    # Ensure the id is an integer and format it
    return "{:05}".format(int(id))

# processed_dataset_save_location = os.path.join(csv_directory, '../datasets/priors_pvas_dataset.pth')
processed_dataset_save_location = os.path.join(save_dir, save_name, '.pth')
image_statistics_pre = []
image_statistics_post = []
no_study_type = []
bad_study_type = []
multiple_entries = []

def pvas_preprocess_image(image, side, image_type, view):
    """
    Load and preprocess the given image. Note that GEO originally had code to also work on different pixel sizes but we didnt need to do that.
    You may need to modify this
    """
    ## Read the dicom and fetch the pixel array
    # image = pydicom.read_file(image_path).pixel_array

    ## Right side mammograms are flipped
    if side == 'R':
        image = np.fliplr(image)

    ## Find otsu cutoff threshold
    cut_off = filters.threshold_otsu(image)

    ## For RAW, apply otsu's, log and invert
    if image_type == 'raw':
        image = np.clip(image, 0, cut_off)
        image = np.log(image+1)
        image = np.amax(image) - image

    ## For PROC, apply otsu's
    elif image_type == 'processed':
        image = np.clip(image, cut_off, np.amax(image))
        image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))

    else:
        print("incorrect type")

    ## Pad images to the same size before resizing
    padded_image = np.zeros((np.amax([2995, image.shape[0]]), np.amax([2394, image.shape[1]])))
    padded_image[0:image.shape[0], 0:image.shape[1]] = image[:, :]
    image = padded_image[0:2995, 0:2394]

    ## Resize to this precise dimension
    image = resize(image, (640, 512))

    ## Max min normalise
    image = image / np.amax(image)

    ## Replicate accross the channels
    image = np.stack((image, image, image), 0)

    ## Normalise to 0 mean 1 s.d. These are PROCAS means (training)
    if view == 'CC':
        if image_type == 'processed':
            norm = T.Normalize((0.147, 0.147, 0.147), (0.261, 0.261, 0.261))
        elif image_type == 'raw':
            norm = T.Normalize((0.183, 0.183, 0.183), (0.331, 0.331, 0.331))
    elif view == 'MLO':
        if image_type == 'processed':
            norm = T.Normalize((0.185, 0.185, 0.185), (0.275, 0.275, 0.275))
        elif image_type == 'raw':
            norm = T.Normalize((0.216, 0.216, 0.216), (0.331, 0.331, 0.331))
    else:
        print("incorrect view")

    image = torch.as_tensor(image.copy())
    image = norm(image)
    return image[0]

def pre_process_mammograms_n_ways(mammographic_images, sides, heights, widths, image_types, process_type='standard'):
    '''
    :param process_type:
        - standard = original version
        - global = rescaled based on global maximum
        - histo = histogram equalisation
        - clahe = Contrast Limited Adaptive Histogram Equalization
    '''
    processed_images = []
    print("Beginning processing", process_type, "images")
    # print(heights, widths)
    for idx, mammographic_image in enumerate(tqdm(mammographic_images)):
        # Extract parameters for each image
        side = sides[idx]
        height = heights[idx]
        width = widths[idx]
        image_type = image_types[idx]

        # Reshape and preprocess
        if side == 'R':
            mammographic_image = np.fliplr(mammographic_image)
        if image_type == 'raw':
            cut_off = mammographic_image > filters.threshold_otsu(mammographic_image)
            cut_off = cut_off.astype(float)
            mammographic_image = cut_off * mammographic_image
            mammographic_image = np.log(mammographic_image+1)
            mammographic_image = np.amax(mammographic_image) - mammographic_image
            if process_type == 'histo':
                mammographic_image = exposure.equalize_hist(mammographic_image, mask=cut_off)
            elif process_type == 'clahe':
                mammographic_image = 2.0 * (mammographic_image - np.min(mammographic_image)) / np.ptp(mammographic_image) - 1
                mammographic_image = exposure.equalize_adapthist(mammographic_image, clip_limit=0.03)
        padded_image = np.zeros((max(2995, mammographic_image.shape[0]), max(2394, mammographic_image.shape[1])))
        padded_image[:mammographic_image.shape[0], :mammographic_image.shape[1]] = mammographic_image
        mammographic_image = resize(padded_image[:2995, :2394], (10 * 64, 8 * 64))
        mammographic_image = mammographic_image / np.amax(mammographic_image)
        processed_images.append(mammographic_image)
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)

def process_images(parent_directory, patient_dir):
    dataset_entries = {p_t: [] for p_t in process_types}

    # patient_path = os.path.join(parent_directory, patient_dir)
    patient_path = patient_dir
    image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

    # Load all images for the given patient/directory
    dcm_files = [pydicom.dcmread(os.path.join(patient_path, f), force=True) for f in image_files]
    if not all([hasattr(dcm, 'StudyDescription') for dcm in dcm_files]):
        print("StudyDescription attribute missing")
        no_study_type.append([patient_dir])#, dcm_files])
        return None
    else:
        studies = [dcm.StudyDescription for dcm in dcm_files]
        if any(s != 'Breast Screening' and s != 'Breast screening' and s != 'BREAST SCREENING' and
               s != 'XR MAMMOGRAM BILATERAL' and s != 'MAMMO 1 VIEW RT' and
               s != 'BILATERAL MAMMOGRAMS 2 VIEWS' for s in studies):
            print("Skipped because not all breast screening - studies =", studies)
            bad_study_type.append([patient_dir, studies])#, dcm_files])
            return None
    print("Appropriate study exists. Processing continuing.")
    all_images = [dcm.pixel_array for dcm in dcm_files]
    for im in all_images:
        if np.sum(np.isnan(im)) > 0:
            print("\nImage was corrupted by", np.sum(np.isnan(im)), "pixel(s)\n")
            return None
    all_sides = ['L' if 'LCC' in f or 'LMLO' in f or 'LSIO' in f else 'R' for f in image_files]
    all_views = ['CC' if 'CC' in f else 'MLO' for f in image_files]
    all_heights = [img.shape[0] for img in all_images]
    if any(num > 4000 for num in all_heights):
        return None
    all_widths = [img.shape[1] for img in all_images]
    if any(num > 4000 for num in all_widths):
        return None
    all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC' not in patient_path)
                       else 'processed' for _ in image_files]

    patient_id = int(patient_path[-5:])
    if np.sum(cancer_data[image_type_id] == patient_id) > 1:
        print(f"\n\n {image_type_id} flagger patient ID {patient_id} as having "
              f"{np.sum(cancer_data[image_type_id] == patient_id)} entries.\nCurrently just the first is taken\n\n")
        multiple_entries.append(patient_id)
        location = cancer_data[image_type_id] == patient_id
        allow_first = True
        for i, idx in enumerate(location):
            if idx:
                if allow_first:
                    allow_first = False
                    continue
                else:
                    location[i] = False
        patient_row = cancer_data[location]
    else:
        patient_row = cancer_data[cancer_data[image_type_id] == patient_id]
    for process_type in process_types:
        # copying to allow processing of the image multiple times
        copied_images = deepcopy(all_images)
        preprocessed_images = torch.stack([pvas_preprocess_image(im, side, type, view) for
                                   im, side, type, view in
                                   zip(copied_images, all_sides, all_image_types, all_views)]).to(torch.float32)
        for im in preprocessed_images:
            if torch.sum(torch.isnan(im)) > 0:
                print("\nImage was corrupted in processing by", torch.sum(np.isnan(im)), "pixel(s)\n")
                return None
        for p_i, i_f, v, s in zip(preprocessed_images, image_files, all_views, all_sides):
            image_data = {
                'view': v,
                'side': s,
                'DCIS': patient_row['DCIS'].item(),
                'IDC': patient_row['IDC'].item(),
                'LCIS': patient_row['LCIS'].item(),
                'Metastatic': patient_row['Metastatic'].item(),
                'Mucinous': patient_row['Mucinous'].item(),
                'Phyllodes': patient_row['Phyllodes'].item(),
                'Papillary': patient_row['Papillary'].item(),
                'Apocrine': patient_row['Apocrine'].item(),
                'Adenoid Cystic': patient_row['Adenoid Cystic'].item(),
                'Metaplastic': patient_row['Metaplastic'].item(),
                'Medullary': patient_row['Medullary'].item(),
                'Tubular': patient_row['Tubular'].item(),
                'ILC': patient_row['ILC'].item(),
                'Invasive Cribriform': patient_row['Invasive Cribriform'].item(),
                'DNK': patient_row['DNK'].item(),
                'no_cancer': patient_row['no_cancer'].item(),
            }
            # if patient_row['no_cancer'].item() == 1:
            #     print("There are non cancers")
            dataset_entries[process_type].append((p_i,
                                                  image_data,
                                                  patient_dir, i_f))
        del copied_images
    del all_images
    del dcm_files
    gc.collect()

    return dataset_entries

# This function will preprocess and zip all images and return a dataset ready for saving
def preprocess_and_zip_all_images(parent_directory):
    dataset_entries = {p_t: [] for p_t in process_types}


    all_dirs = []
    print("Collecting directories")
    for root, dirs, _ in tqdm(os.walk(parent_directory)):
        for d in dirs:
            if d[-5:] == '_anon':
                print("anon folder found")
                # for root2, dirs2, _ in os.walk(os.path.join(root, d)):
                #     for d2 in dirs2:
                #         dir_path = os.path.join(root2, d2)
                #         all_dirs.append(dir_path)
            else:
                dir_path = os.path.join(root, d)
                all_dirs.append(dir_path)

    patient_dirs = [d for d in all_dirs if int(d[-5:]) in np.array(cancer_data[image_type_id], dtype=int)]
    patient_dirs.sort()  # Ensuring a deterministic order


    for p_i, patient_dir in enumerate(tqdm(patient_dirs)):
        print("Processing", p_i, "/", len(patient_dirs), "of", save_name, "for patient", patient_dir)
        processed_images = process_images(parent_directory, patient_dir)
        if processed_images == None:
            continue
        for process_type in dataset_entries:
            for stuff in processed_images[process_type]:
                dataset_entries[process_type].append(stuff)

    return dataset_entries

if __name__ == "__main__":
    # Generate the dataset and save it
    # if not os.path.exists(processed_dataset_save_location):
    if not raw or creating_pvas_loader:
        process_types = ['base']
    # tracemalloc.start()
    dataset_entries = preprocess_and_zip_all_images(image_directory)
    for process_type in dataset_entries:
        # torch.save(dataset_entries[process_type], processed_dataset_save_location[:-4]+'_'+process_type+'_otsu_1st.pth')
        save_location_and_name = processed_dataset_save_location[:-5]+'_'+process_type+'.pth'
        print("Saving to", save_location_and_name, "of length", len(dataset_entries[process_type]))
        torch.save(dataset_entries[process_type], save_location_and_name)
        # torch.save(dataset_entries[process_type], 'C:/Users/adam_/PycharmProjects/pVAS/datasets/priors_pvas_dataset.pth')

    print("Patients with multiple entries", multiple_entries)

    print("Done")

