import matplotlib.pyplot as plt
import numpy as np
from csv import DictReader
from sklearn.metrics import mean_squared_error
import os
import pandas as pd

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

x, y = [], []

csv_folder = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'

adam_predictions = pd.read_csv(os.path.join(csv_folder, "all_cross_fold_data_adam.csv"))

alistair_predictions = pd.read_csv(os.path.join(csv_folder, "all_cross_fold_data_new_alistair.csv"))
    
# testing = pd.read_csv(os.path.join(csv_folder, "_vendors_grouped_pilot_50subjects_rearranged.csv"))
testing = pd.read_csv(os.path.join(csv_folder, "_vendors_grouped_Reader_1704subjects.csv"))

# 1. Convert column data types for merging
adam_predictions['case'] = adam_predictions['case'].astype(str)
alistair_predictions['case'] = alistair_predictions['case'].astype(str)
testing['Case'] = testing['Case'].astype(str)

adam_predictions['timepoint'] = adam_predictions['timepoint'].astype(float)
alistair_predictions['timepoint'] = alistair_predictions['timepoint'].astype(float)
testing['TimePoint'] = testing['TimePoint'].astype(float)

# 2. Merge adam and alistair predictions on 'case' and 'timepoint'
combined_predictions = pd.merge(
    adam_predictions,
    alistair_predictions,
    on=['case', 'timepoint'],
    suffixes=('_adam', '_alistair')
)

pilot_data = []
adam_data = {'mean': [], 'std': []}
alistair_data = {'mean': [], 'std': []}
combined_data = {'mean': [], 'std': []}
combined_and_weighted_data = {'mean': [], 'std': []}
for index, row in testing.iterrows():
    case = str(row['Case'])
    tp = float(row['TimePoint'])
    ave_reading = row['VAS10']
    # ave_reading = row['Score']
    adam_condition = (adam_predictions['case'] == case) & (adam_predictions['timepoint'] == tp)
    alistair_condition = (alistair_predictions['case'] == int(case)) & (alistair_predictions['timepoint'] == int(tp))
    if len(adam_predictions.loc[adam_condition]) < 1 or len(alistair_predictions.loc[alistair_condition]) < 1:
        if len(adam_predictions.loc[adam_condition]) < 1:
            print(f"{len(adam_predictions.loc[adam_condition])}failed for Adam during case {case} and tp {tp}")
        if len(alistair_predictions.loc[alistair_condition]) < 1:
            print(f"{len(alistair_predictions.loc[alistair_condition])}failed for Alistair during case {case} and tp {tp}")
        continue
    pilot_data.append(ave_reading)
    adam_data['mean'].append(float(adam_predictions.loc[adam_condition, 'mean'].values[0]))
    alistair_data['mean'].append(float(alistair_predictions.loc[alistair_condition, 'mean'].values[0]))
    combined_data['mean'].append((adam_data['mean'][-1] + alistair_data['mean'][-1]) / 2)
    adam_count = float(alistair_predictions.loc[adam_condition, 'count'].values[0])
    alistair_count = float(alistair_predictions.loc[alistair_condition, 'count'].values[0])
    combined_and_weighted_data['mean'].append(((adam_data['mean'][-1] * adam_count) + (alistair_data['mean'][-1] * alistair_count)) / (adam_count + alistair_count))
    adam_data['std'].append(float(adam_predictions.loc[adam_condition, 'std'].values[0]))
    alistair_data['std'].append(float(alistair_predictions.loc[alistair_condition, 'std'].values[0]))
    combined_data['std'].append((adam_data['std'][-1] + alistair_data['std'][-1]) / 2)
    combined_and_weighted_data['std'].append(((adam_data['std'][-1] * adam_count) + (alistair_data['std'][-1] * alistair_count)) / (adam_count + alistair_count))

plt.figure(figsize=(8, 6))
plt.errorbar(pilot_data, adam_data['mean'], yerr=adam_data['std'], fmt='o', label='Adam', capsize=3)
plt.errorbar(pilot_data, alistair_data['mean'], yerr=alistair_data['std'], fmt='o', label='Alistair', capsize=3)
plt.plot([0, 100], [0, 100], '--', lw=2, color='red')
plt.xlabel('pilot')
plt.ylabel('prediction')
plt.title('Scatter Plot with Error Bars')
plt.legend()
plt.show()

pilot_predictions = np.array(pilot_data)
# adam_mean = np.array(adam_data['mean'])
# alistair_mean = np.array(alistair_data['mean'])
adam_mean = np.array(combined_and_weighted_data['mean'])
alistair_mean = np.array(alistair_data['mean'])
# Bland-Altman calculations
mean_values = (adam_mean + alistair_mean) / 2
differences = adam_mean - alistair_mean
std_diff = np.std(differences)
mean_diff = np.mean(differences)
mse = np.mean(np.square(differences))

# Plot
plt.figure(figsize=(8, 6))

# Scatter plot for Bland-Altman
plt.scatter(mean_values, differences, label='Differences', color='blue')

# Add limits of agreement
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean Difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')

# Labels and legend
plt.xlabel('Mean of Adam and Alistair')
plt.ylabel('Difference (Adam - Alistair)')
plt.xlim([0, 100])
plt.ylim([-25, 25])
plt.title('Bland-Altman Plot Adam vs Alistair - MSE: {:.2f}'.format(mse))
plt.legend()
plt.show()
# Bland-Altman calculations
mean_values = (adam_mean + pilot_predictions) / 2
differences = adam_mean - pilot_predictions
std_diff = np.std(differences)
mean_diff = np.mean(differences)
mse = np.mean(np.square(differences))

# Plot
plt.figure(figsize=(8, 6))

# Scatter plot for Bland-Altman
plt.scatter(mean_values, differences, label='Differences', color='blue')

# Add limits of agreement
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean Difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')

# Labels and legend
plt.xlabel('Mean of Adam and Pilot')
plt.ylabel('Difference (Adam - Pilot)')
plt.xlim([0, 100])
plt.ylim([-25, 25])
plt.title('Bland-Altman Plot Adam vs Pilot - MSE: {:.2f}'.format(mse))
plt.legend()
plt.show()
# Bland-Altman calculations
mean_values = (pilot_predictions + alistair_mean) / 2
differences = alistair_mean - pilot_predictions
std_diff = np.std(differences)
mean_diff = np.mean(differences)
mse = np.mean(np.square(differences))

# Plot
plt.figure(figsize=(8, 6))

# Scatter plot for Bland-Altman
plt.scatter(mean_values, differences, label='Differences', color='blue')

# Add limits of agreement
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean Difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')

# Labels and legend
plt.xlabel('Mean of Alistair and pilot')
plt.ylabel('Difference (Alistair - pilot)')
plt.xlim([0, 100])
plt.ylim([-25, 25])
plt.title('Bland-Altman Plot Alistair vs Pilot - MSE: {:.2f}'.format(mse))
plt.legend()
plt.show()

print("done")
