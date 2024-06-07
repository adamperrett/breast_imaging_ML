import pandas as pd
import numpy as np
from tqdm import tqdm

# Load your data
df = pd.read_csv('../csv_data/outlier_processing.csv')

views = ['L', 'R']
positions = ['CC', 'MLO']
readers = ['-1', '-2']
outlier_type = 'm_median_dist'


def save_csv(df, method, threshold):
    filename = f'outlier_processed_by_{method}_threshold_{threshold}.csv'
    print('Create combined readings by averaging the adjusted reader values')
    df_combined = df.copy()
    for view in views:
        for pos in positions:
            df_combined[f'{view}{pos}'] = df[[f'{view}{pos}-1', f'{view}{pos}-2']].mean(axis=1)
    df_combined.to_csv('../csv_data/'+filename, index=False)


for threshold in tqdm([2, 6, 10, 14, 18, 22, 26, 30]):
    df_copy = df.copy()

    print('Identify outliers based on the threshold for both views')
    for view in views:
        for pos in positions:
            df_copy[f'{view}{pos}_is_outlier'] = ((df_copy[f'{view}{pos}-1_{outlier_type}'] >= threshold) |
                                                   (df_copy[f'{view}{pos}-2_{outlier_type}'] >= threshold))

    print('Remove by patient')
    df_remove_patient = df_copy[~df_copy.apply(
        lambda x: x['LCC_is_outlier'] or x['RCC_is_outlier'] or x['LMLO_is_outlier'] or x['RMLO_is_outlier'],
        axis=1)]
    save_csv(df_remove_patient, 'remove_patient', threshold)

    df_remove_image = df_copy.copy()
    for view in views:
        for pos in positions:
            for reader in readers:
                # Remove that image only
                df_remove_image.loc[
                    df_copy[f'{view}{pos}{reader}_{outlier_type}'] >= threshold, f'{view}{pos}{reader}'] = np.nan

    save_csv(df_remove_image, 'remove_image', threshold)

    print('Replace outlier value with an average of all other readings for that breast')
    df_replace_avg = df_copy.copy()
    for view in views:
        other_readings = df_copy[[f'{view}MLO-1', f'{view}MLO-2',
                                  f'{view}CC-1', f'{view}CC-2']].copy()
        for pos in positions:
            other_readings[f'{view}{pos}-1'] = np.where(
                df_copy[f'{view}{pos}-1_{outlier_type}'] >= threshold, np.nan, df_copy[f'{view}{pos}-1'])
            other_readings[f'{view}{pos}-2'] = np.where(
                df_copy[f'{view}{pos}-2_{outlier_type}'] >= threshold, np.nan, df_copy[f'{view}{pos}-2'])

            avg_readings = other_readings.mean(axis=1)

            df_replace_avg[f'{view}{pos}-1'] = np.where(
                df_copy[f'{view}{pos}-1_{outlier_type}'] >= threshold, avg_readings, df_copy[f'{view}{pos}-1'])
            df_replace_avg[f'{view}{pos}-2'] = np.where(
                df_copy[f'{view}{pos}-2_{outlier_type}'] >= threshold, avg_readings, df_copy[f'{view}{pos}-2'])
    save_csv(df_replace_avg, 'replace_with_avg', threshold)

    # print('Replace with the other reader score for that view')
    # df_replace_reader = df_copy.copy()
    # for view in views:
    #     for pos in positions:
    #         df_replace_reader[f'{view}{pos}-1'] = np.where(
    #             df_copy[f'{view}{pos}-1_{outlier_type}'] >= threshold,
    #             df_copy[f'{view}{pos}-2'],
    #             df_copy[f'{view}{pos}-1'])
    #         df_replace_reader[f'{view}{pos}-2'] = np.where(
    #             df_copy[f'{view}{pos}-2_{outlier_type}'] >= threshold,
    #             df_copy[f'{view}{pos}-1'],
    #             df_copy[f'{view}{pos}-2'])
    #
    # save_csv(df_replace_reader, 'replace_reader', threshold)
