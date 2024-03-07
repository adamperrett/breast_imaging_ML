import sys
import os


num_epochs = 600
patience = 150
lr = 0.003
momentum = 0.9
batch_size = 16
op_choice = 'adam'

weighted = 0
transformed = 0

raw = True
pvas_loader = False
split_CC_and_MLO = True
CC_or_MLO = 'CC'

by_patient = False
if by_patient:
    parallel_images = 8
else:
    parallel_images = 0


on_CSF = True

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
    configurations = []
    for n_im in [8, 4, 2, 1]:
        for b_size in [128, 64, 32, 24, 16, 8]:
            for op_choice in ['adam', 'sgd']:
                for d_set in ['proc', 'log', 'histo', 'clahe']:
                    for weight_choice in [0, 1]:
                        for trans_choice in [0, 1]:
                            configurations.append({
                                'dataset': d_set,
                                'batch_size': b_size,
                                'optimizer': op_choice,
                                'weighted': weight_choice,
                                'transformed': trans_choice,
                                'n_images': n_im
                            })
    config = int(sys.argv[1]) - 1

    processed = True
    dataset_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast-cancer/'
    config = configurations[config]
    processed_dataset_path = os.path.join(dataset_dir,
                                          'mosaics_processed/full_mosaic_dataset_{}.pth'.format(config['dataset']))
    batch_size = config['batch_size']
    op_choice = config['optimizer']
    weighted = config['weighted']
    transformed = config['transformed']
    n_images = config['n_images']

    working_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast_imaging_ML/training/'
    best_model_name = 'VAS_csf_{}_{}_{}x{}_t{}_w{}_js{}'.format(
        config['dataset'], op_choice, batch_size, n_images, transformed, weighted, int(sys.argv[1]))

    print("Config", int(sys.argv[1]) + 1, "creates test", best_model_name)
else:

    n_images = 8

    image_directory = 'D:/mosaic_data/raw'
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast-cancer/data'
    csv_name = 'full_procas_info3.csv'
    reference_csv = 'PROCAS_reference.csv'

    keyword = 'local_testing'
    dataset = 'log'

    processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/mosaic_pvas_dataset_{}.pth'.format(dataset))
    # processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset_log.pth')

    working_dir = 'C:/Users/adam_/PycharmProjects/breast-cancer/training/'
    best_model_name = '{}_{}_{}_{}x{}_t{}_w{}'.format(
        keyword, dataset, op_choice, batch_size, n_images, transformed, weighted)