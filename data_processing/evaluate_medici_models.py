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

testing_data = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data/local_pvas_vbd_processed_per_im_base.pth'

model_location = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/models/medici_models'
# model_name = 'l_medici_testing_lr5.93e-05x23_34mean_p1r0_d0.236_adam_t0_wl0_ws0'
model_name = 'l_medici_testing_lr5.87e-05x6_50mean_p1r0_d0.196_adam_t0_wl0_ws0'
save_location = model_location
test_name = model_name
split = 0
pre_trained = 1
resnet_size = 50
pooling_type = 'mean'
replicate = 0
dropout = 0.159
batch_size = 32

num_manufacturers = 8
manufacturer_mapping = {'Philips': nn.functional.one_hot(torch.tensor(0),
                                                         num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'SIEMENS': nn.functional.one_hot(torch.tensor(1),
                                                         num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'HOLOGIC': nn.functional.one_hot(torch.tensor(2),
                                                         num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'GE': nn.functional.one_hot(torch.tensor(3),
                                                    num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'KODAK': nn.functional.one_hot(torch.tensor(4),
                                                       num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'FUJIFILM': nn.functional.one_hot(torch.tensor(5),
                                                          num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'IMS': nn.functional.one_hot(torch.tensor(6),
                                                     num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                        'LORAD': nn.functional.one_hot(torch.tensor(7),
                                                       num_classes=num_manufacturers).to(torch.float32).to('cuda')
                        }

print('Loading model')
model = Medici_MIL_Model(pre_trained, replicate, resnet_size, pooling_type, dropout,
                         split=split_CC_and_MLO, num_manufacturers=num_manufacturers).to('cuda')
# model = Pvas_Model(pretrain=False, replicate=replicate, dropout=dropout).to('cuda')
model.load_state_dict(torch.load(os.path.join(model_location, model_name)))

print('Making dataloader')
training_mean, training_std = 6.448529270685709, 4.314404263091535
pilot_loader = return_medici_loaders(processed_pilot_file, batch_size=batch_size, only_testing=True,
                                     transformed=False, weighted_loss=False, weighted_sampling=False)

print("Evaluating model")
loss, labels, preds, r2, r2w, error, conf_int, manufacturers = evaluate_medici(
    model, pilot_loader, criterion,
    inverse_standardize_targets, training_mean, training_std,
    num_manufacturers=num_manufacturers,
    manufacturer_mapping=manufacturer_mapping,
    return_names=False, split_CC_and_MLO=split)
print("loss:", loss)
print("r2:", r2)

print("Splitting into different views and files")

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
# csv_name = 'volpara_priors_testing_weight.csv'
# csv_name = 'PROCAS_matched_priors_v2.csv'
# csv_name = '_vendors_grouped_pilot_50subjects_rearranged.csv'
# csv_name = 'pilot_and_AI_prediction.csv'
csv_name = 'VAS_readings_all.csv'
pilot_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

if len(preds) == len(pilot_data):
    pilot_data[model_name] = preds  # Add the list as a new column with the column name from model_name
else:
    print("The length of preds does not match the number of rows in the DataFrame.")

# Save the updated dataframe to a new CSV file
output_csv_name = 'pilot_and_AI_prediction.csv'  # Update this filename as needed
pilot_data.to_csv(os.path.join(csv_directory, output_csv_name), index=False)

print(f"Updated data saved to {os.path.join(csv_directory, output_csv_name)}")

columns_to_plot = ['aevans', 'emuscat', 'mtelesca', 'ssavaridas', 'pwhelehan',
                   'smuthyala', 'sdrummond', 'nhealy', 'asharma', 'svinnicombe', 'jnash', model_name,
                   'VAS10', 'VAS11', 'R1_inVAS', 'R2_inVAS', 'R3_inVAS', 'R4_inVAS', 'ave_us']

# Get the 'Average' and 'Manufacturer' columns
average_values = pilot_data['VAS10']
manufacturers = pilot_data['Manufacturer']

# Get unique manufacturers for coloring
unique_manufacturers = np.unique(manufacturers)

# Generate a colormap for manufacturers
colors = plt.cm.get_cmap('tab10', len(unique_manufacturers))  # A discrete colormap with 10 colors
manufacturer_color_map = {manufacturer: colors(i) for i, manufacturer in enumerate(unique_manufacturers)}

# Loop over each column (except 'Average' and 'Manufacturer') and create scatter plots
mse_reader_vs_manufacturer = {}
for column in columns_to_plot:
    fig, ax = plt.subplots(figsize=(10, 6))
    mse_reader_vs_manufacturer[column] = {}
    # Scatter plot for each column against the 'Average' column
    for manufacturer in unique_manufacturers:
        # Mask to filter data by manufacturer
        mask = manufacturers == manufacturer
        ax.scatter(average_values[mask], pilot_data[column][mask], label=manufacturer,
                   color=manufacturer_color_map[manufacturer], alpha=0.6)
        manu_mse = np.sum(np.square(average_values[mask] - pilot_data[column][mask])) / np.sum(mask)
        print(f"MSE for {column} on manufacturer {manufacturer} is {manu_mse}")
        mse_reader_vs_manufacturer[column][manufacturer] = manu_mse

    # Set labels and title
    ax.set_xlabel('Average')
    ax.set_ylabel(column)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.plot([0, 100], [0, 100], '--', lw=2, color='red')
    mse = np.sum(np.square(average_values - pilot_data[column])) / len(average_values)
    mse_reader_vs_manufacturer[column]['overall'] = mse
    print(f"MSE for {column} is {mse}")
    # ax.set_title(f'{column} vs Average - MSE:{mse}')

    # Add legend for manufacturers
    ax.legend(title="Manufacturer")

    # Show the plot or save it if needed
    plt.savefig(f"{save_location}/{column}_vs_Average.png", bbox_inches='tight', format='png')
    plt.close()

mse_df = pd.DataFrame(mse_reader_vs_manufacturer)

# Save the DataFrame to a CSV file
csv_save_path = os.path.join(save_location, 'mse_reader_vs_manufacturer_all.csv')  # Adjust your path as needed
mse_df.to_csv(csv_save_path)

print("Done")
