import sys
import os


num_epochs = 600
patience = 5
lr = 0.003
momentum = 0.9
# batch_size = 128
op_choice = 'adam'

weight_samples = 1
weight_loss = 0
transformed = 0

raw = False
pvas_loader = False
split_CC_and_MLO = False
CC_or_MLO = 'CC'

by_patient = False
if by_patient:
    parallel_images = 8
else:
    parallel_images = 0


on_CSF = True
optuna_optimisation = True

if on_CSF:
    '''
    Test across:
    -n images 4
    -dataset 4
    -batch_size 6
    -optimiser 2
    -weights 2
    -transformed 2
    '''
    working_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/training/'
    # working_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/training/'
    if optuna_optimisation:
        base_name = 'volpara_weight'
        n_images = 0
        # processed_dataset_path = '/mnt/bmh01-rds/assure/processed_data/'
        processed_dataset_path = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/processed_data/'
        # processed_dataset_path = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/'
        data_name = 'procas_pvas_vbd_processed_per_im_base'
        # data_name = 'priors_pvas_vbd_processed_per_im_base'
        if split_CC_and_MLO:
            base_name += CC_or_MLO
            if CC_or_MLO == 'CC':
                data_name += '_CC'
            else:
                data_name += '_MLO'
        processed_dataset_path += data_name+'.pth'

        # data_name = 'priors_pvas_processed_base_CC'
        # processed_dataset_path = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/priors_pvas_processed_base_CC.pth'
    else:
        configurations = []
        for b_size in [512, 256, 128, 64]:
            for op_choice in ['adam', 'sgd', 'd_adam', 'd_sgd']:
                for weight_choice in [0, 1]:
                    for trans_choice in [0, 1]:
                        configurations.append({
                            'lr': lr,
                            'batch_size': b_size,
                            'optimizer': op_choice,
                            'weighted': weight_choice,
                            'transformed': trans_choice
                        })
        best_model_name = 'VAS_csf_{}_{}x{}_t{}_w{}_{}'.format(
            op_choice, batch_size, lr, transformed, weighted, int(sys.argv[1]))

        print("Config", int(sys.argv[1]) + 1, "creates test", best_model_name)
else:

    n_images = 8

    image_directory = 'D:/mosaic_data/raw'
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast-cancer/data'
    csv_name = 'full_procas_info3.csv'
    reference_csv = 'PROCAS_reference.csv'

    keyword = 'local_testing'
    dataset = 'log'

    # processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/mosaic_pvas_dataset_{}.pth'.format(dataset))
    # processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset_log.pth')

    working_dir = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/training/'
    best_model_name = '{}_{}_{}_{}x{}_t{}_w{}'.format(
        keyword, dataset, op_choice, batch_size, n_images, transformed, weighted)