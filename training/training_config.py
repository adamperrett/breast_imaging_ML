import sys


num_epochs = 600
patience = 10
improving_loss_or_r2 = 'auc'
r2_weighting_offset = 0
lr = 0.003
momentum = 0.9
# batch_size = 128
op_choice = 'adam'

weight_samples = 1
weight_loss = 0
transformed = 0

raw = True
pvas_loader = False
split_CC_and_MLO = False
CC_or_MLO = 'CC'

by_patient = False
if by_patient:
    parallel_images = 8
else:
    parallel_images = 0

medici_crossval = True
mosaics_processing = False
medici_processing = True
combined_processing = True
on_CSF = True
optuna_optimisation = True
recurrence_optimisation = True
CRUK_optimisation = True

if on_CSF:
    # data_name = 'local_pvas_vas_raw_base'
    # working_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/training/'
    # processed_dataset_path = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/'
    working_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/training/'
    processed_dataset_path = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data/'
    if CRUK_optimisation:
        data_name = 'medici_classification_preprocessed_data'
        base_name = 'DCIS_IDC'
    elif recurrence_optimisation:
        data_name = 'medici_classification_preprocessed_data'
        base_name = 'focal_sample_52525'
    elif mosaics_processing:
        if combined_processing:
            data_name_2 = 'raw_mosaic_dataset_log'
            # data_name_1 = 'procas_pvas_vas_raw_base'
            data_name_1 = 'priors_pvas_vas_raw_base'
        data_name = 'raw_mosaic_dataset_log'
        base_name = 'non_mosaic_mil_testing'
    elif medici_crossval:
        data_name = 'medici_5_vendors_preprocessed_data_combined'
        base_name = 'medici_crossval'
    elif medici_processing:
        data_name = 'medici_preprocessed_data'
        base_name = 'medici_testing'
    else:
        data_name = 'procas_pvas_vas_raw_base'
        base_name = 'vas_outlier_reseeding'
    # processed_dataset_path = '/mnt/bmh01-rds/assure/processed_data/'
    # data_name = 'procas_pvas_vbd_processed_per_im_base'
    if medici_processing:
        pilot_name = 'medici_preprocessed_pilot'
        if split_CC_and_MLO:
            pilot_name += '_' + CC_or_MLO
    else:
        priors_name = 'priors_pvas_vas_raw_base'
        if split_CC_and_MLO:
            priors_name += '_' + CC_or_MLO
    if split_CC_and_MLO:
        base_name += '_' + CC_or_MLO
        data_name += '_' + CC_or_MLO
    if medici_processing:
        processed_pilot_file = pilot_name
    else:
        processed_priors_file = priors_name
    processed_dataset_file = data_name
else:


    # working_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/training/'
    working_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/training/'
    base_name = 'vas_baseline'
    # processed_dataset_path = '/mnt/bmh01-rds/assure/processed_data/'
    # processed_dataset_path = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data/'
    processed_dataset_path = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/'
    # data_name = 'procas_pvas_vbd_processed_per_im_base'
    # data_name = 'local_pvas_vas_raw_base'
    # data_name = 'procas_pvas_vas_raw_base'
    if raw:
        data_name = 'CRUK_local_raw_base'
    else:
        data_name = 'CRUK_local_processed_base'
    priors_name = 'priors_pvas_vas_raw_base'
    if split_CC_and_MLO:
        base_name += '_' + CC_or_MLO
        data_name += '_' + CC_or_MLO
        priors_name += '_' + CC_or_MLO
    processed_priors_file = priors_name
    processed_dataset_file = data_name