import numpy as np
import pandas as pd
import os
import datetime as dt
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
save_dir = csv_directory  # 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'

csv_name = 'PROCAS_full_data_16-03-2024.csv'
save_name = 'processed_PROCAS_full_data_16-03-2024.csv'

csv_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')

csv_processed_previously = True

def date_to_number(d):
    # expected format "23/06/2025"
    if not d or not isinstance(d, str) or len(d) < 10:
        return None
    try:
        return dt.date(int(d[6:]), int(d[3:5]), int(d[:2])).toordinal()
    except Exception:
        return None

def number_to_date(n):
    if n is None:
        return ""
    try:
        d = dt.date.fromordinal(int(n))
        return d.strftime("%d/%m/%Y")
    except Exception:
        return ""

def address_to_gps(address):
    # split by comma
    # split by space
    # use postcode to search csv with coord data
    # return coords
    return "address_as_coords"

def too_many_discrete(category):
    return category

def clean_numbers(numbers, num_type):
    return num_type(numbers)

string_mapping = {
    # general framework
    'column_name': {
        'old_entry': 'replacement',         # n=a
        '': 'replacement_for_blank_entry'   # n=b
    },

    # actual mappings
    'post prev biopsy': {
        '0': 'No',   # n=53462
        '1': 'Yes',   # n=3271
        '2': 'Yes',   # n=17
        '': 'DNK'   # n=1153
    },
    'DiagnosisOfCancer <70': {
        'na': 'No',     # n=15
        'No': 'No',     # n=56140
        'no': 'No',     # n=56140
        'yes': 'Yes',   # n=1747
        'Yes': 'Yes',   # n=1747
        '': 'DNK'       # n=1
    },
    'screen detected': {
        'incident': 'incident',     # n=995
        'interval': 'interval',     # n=475
        'other': 'other',           # n=72
        'prevalent': 'prevalent',   # n=464
        '': "Not detected"          # n=55897
    },
    'FDR OC': {
        'No': 'No',     # n=25
        'Yes': 'Yes',   # n=1356
        '': 'n/a'       # n=56522
    },
    'Chemoprev Drug': {
        'Anastrozole': 'Anastrozole',   # n=76
        'Raloxifene': 'Raloxifene',     # n=140
        'raloxifene': 'Raloxifene',     # n=140
        'tamoxifen': 'Tamoxifen',       # n=100
        '': 'None'                      # n=57587
    },
    'FHC': {
        'no': 'No',     # n=56546
        'yes': 'Yes',   # n=1356
        '': 'DNK'        # n=1
    },
    'PROCAS after': {
        'no': 'No',     # n=716
        'yes': 'Yes',   # n=637
        '': 'n/a'     # n=56547
    },
    'Highmod gene': {
        'ATM': 'ATM',               # n=13
        'BRCA1': 'BRCA1',           # n=10
        'BRCA2': 'BRCA2',           # n=22
        'BRIP1 missense': 'BRIP1',  # n=1
        'CHEK2': 'CHEK2',           # n=14
        'MSH6': 'MSH6',             # n=4
        'NF1': 'NF1',               # n=2
        'no': 'None',               # n=2090
        'PALB2': 'PALB2',           # n=6
        'PALB2 c.3113G>A': 'PALB2', # n=2
        'PTEN': 'PTEN',             # n=1
        'TP53': 'TP53',             # n=1
        '': 'DNK'                   # n=55737
    },
    'panel test': {
        'FHC test': 'FHC test', # n=212
        'no': 'No',             # n=55712
        'yes': 'Yes',           # n=1978
        '': 'n/a'               # n=1
    },
    'Bilateral': {
        'yes': 'Yes',   # n=101
        '': 'No'        # n=57802
    },
    'DC study': {
        'yes': 'Yes',                   # n=1250
        'yeshealthy': 'Healthy',        # n=2904
        'yesobese': 'Obese',            # n=1539
        'yesunderover': 'Underover',    # n=206
        '': 'Blank'                     # n=52004
    },
    'cancer death': {
        'no': 'No',     # n=89
        'yes': 'Yes',   # n=84
        'else': date_to_number,   # n=515 (or should this just be Yes?) todo fix this
        '': 'Blank'     # n=57215
    },
    'alcohol death': {
        'no': 'No',     # n=163
        'yes': 'Yes',   # n=9
        '?': 'Blank',   # n=1
        '': 'Blank'     # n=57730
    },
    'CVD death': {
        'no': 'No',     # n=140
        'yes': 'Yes',   # n=34
        '': 'Blank'     # n=57729
    },
    'SNPs': {
        'yes': 'Yes',   # n=9523
        '': 'Blank'     # n=48380
    },
    'ConsentedToDNA': {
        'No': 'No',     # n=47883
        'Yes': 'Yes',   # n=10019
        '': 'Blank'     # n=1
    },
    'DiagnosisOfCancer': {
        'Yes': 'Yes',   # n=1532
        '': 'No'        # n=56371
    },
    'DiagnosisOfCancer2': {
        'No': 'No',     # n=55746
        'Yes': 'Yes',   # n=2156
        '': 'No'        # n=1
    },
    'Hysterectomy': {
        'No': 'No',     # n=43453
        'Yes': 'Yes',   # n=13943
        'DNK': 'DNK',   # n=498
        '': 'DNK'       # n=9
    },
    'OvariesRemoved': {
        'No': 'No',     # n=42022
        'Yes': 'Yes',   # n=5
        'Both': 'Both', # n=4977
        'One': 'One',   # n=2029
        'DNK': 'DNK',   # n=8868
        '': 'DNK'       # n=2
    },
    'OvarianCancerYN': {
        'Yes': 'Yes',   # n=69
        '': 'No'        # n=57834
    },
    'OvarianCancerYN2': {
        'Y': 'Yes',     # n=1
        'Yes': 'Yes',   # n=71
        '': 'No'        # n=57831
    },
    'AnyChildrenYN': {
        'No': 'No',     # n=7384
        'Yes': 'Yes',   # n=50411
        'DNK': 'DNK',   # n=107
        '': 'DNK'       # n=1
    },
    'within 10-year grp': {
        'yes': 'Yes',   # n=225
        '': 'Blank'     # n=57678
    },
    'Endometrial cancer': {
        'yes': 'Yes',   # n=103
        '': 'Blank'     # n=57800
    },
    'EC or OC': {
        'C54 - Malignant neoplasm of corpus uteri': 'OC',               # n=68
        'C54 - Malignant neoplasm of corpus uteri also Ovary': 'Both',  # n=1
        'C56.X - Malignant neoplasm of ovary': 'OC',                    # n=42
        'C56.X - Malignant neoplasm of ovary and endometrium': 'Both',  # n=1
        'EC': 'EC',                                                     # n=6
        'Endometrial': 'EC',                                            # n=27
        'Ovarian': 'OC',                                                # n=5
        'ovarian cancer': 'OC',                                         # n=1
        '': 'Blank'                                                     # n=57752

    },
    'FDR breast': {
        'no': 'No',     # n=41705
        'yes': 'Yes',   # n=6903
        '': 'Blank',    # n=9295
    },
    'FDR50': {
        'yes': 'Yes',   # n=2164
        '': 'Blank',    # n=55739
    },
    '2FDR': {
        'yes': 'Yes',   # n=890
        '': 'Blank',    # n=57013
    },
    'SDR': {
        'yes': 'Yes',   # n=11184
        '': 'Blank',    # n=46719
    },
    '2+FHno50': {
        'more sig': 'more sig', # n=2200
        'yes': 'Yes',           # n=1350
        '': 'Blank',            # n=54353
    },
    'NICE criteria': {
        'yes': 'Yes',   # n=4664
        '': 'Blank',    # n=53293
    },
    'NICE BC FDR criteria met': {
        'yes': 'Yes',   # n=2351
        '': 'Blank',    # n=55552
    },
    'NICE high risk': {
        'yes': 'Yes',   # n=613
        '': 'Blank',    # n=57290
    },
    'One FDR <40': {
        'yes': 'Yes',   # n=421
        '': 'Blank',    # n=57482
    },
    'Status': {
        'Active': 'Active',     # n=54819
        'Deceased': 'Deceased', # n=3056
        'Lost': 'Lost',         # n=27
        '': 'Blank',            # n=1
    },
    'All good factors': {
        'spades': 'spades', # n=5
        'yes': 'Yes',       # n=28
        '': 'Blank',        # n=57870
    },
    'Wtage20available': {
        'yes': 'Yes',   # n=50791
        '': 'Blank',    # n=7112
    },
    'BMI20<21.6': {
        'no': 'No',     # n=6360
        'yes': 'Yes',   # n=34918
        '': 'Blank',    # n=16625
    },
    'AnyExercise': {
        'DNK': 'DNK',   # n=5716
        'No': 'No',     # n=10801
        'no': 'No',     # n=10801
        'Yes': 'Yes',   # n=41385
        'yes': 'Yes',   # n=41233
        '': 'Blank',    # n=1
    },
    'AlcoholYN': {
        'DNK': 'DNK',   # n=850
        'No': 'No',     # n=15817
        'Yes': 'Yes',   # n=41233
        'yes': 'Yes',   # n=41233
        '': 'Blank',    # n=3
    },
    'MenopausalStatus': {
        'Data not known': 'DNK',            # n=3071
        'Datanot known': 'DNK',             # n=4
        'Not applicable': 'n/a',            # n=1
        'perimenopausal': 'perimenopausal', # n=10720
        'postmenopausal': 'postmenopausal', # n=37233
        'premenopausal': 'premenopausal',   # n=6873
        '': 'Blank',                        # n=1
    },
    'postmen': {
        'no': 'No',     # n=20669
        'yes': 'Yes',   # n=37233
        '': 'Blank',    # n=1
    },
    'HRT': {
        'DNK': 'DNK',   # n=540
        'No': 'No',     # n=36508
        'Yes': 'Yes',   # n=20854
        '': 'Blank',    # n=1
    },
    'HRT2': {
        'DNK': 'DNK',   # n=540
        'No': 'No',     # n=36508
        'Yes': 'Yes',   # n=20854
        '': 'Blank',    # n=1
    },
    'Combined HRT': {
        'unknown': 'DNK',   # n=12286
        'No': 'No',         # n=4008
        'yes': 'Yes',       # n=4567
        'else': 'No',       # n=37041 (a bunch of random patient IDs)
        '': 'Blank',        # n=1
    },
    'HRTType': {
        'Data not known': 'DNK',            # n=4566
        'Combined': 'Combined',             # n=1194
        'Oestrogen only': 'Oestrogen only', # n=4007
        '': 'Blank',                        # n=48136
    },
    'HRT pre50': {
        'no': 'No',     # n=17297
        'yes': 'Yes',   # n=3556
        '': 'Blank',    # n=37050
    },
    'HRT Current?': {
        'Data not known': 'DNK',    # n=13
        'No': 'No',                 # n=51810
        'Yes': 'Yes',               # n=4418
        '': 'Blank',                # n=1662
    },
    'HRT recent': {
        'no': 'No',     # n=9090
        'yes': 'Yes',   # n=11488
        '': 'Blank',    # n=37325
    },
    'EthnicOrigin': {
        'Asian or Asian British': 'Asian or Asian British', # n=891
        'Black or Black British': 'Black or Black British', # n=671
        'Data not known': 'DNK',                            # n=1867
        'Jewish': 'Jewish',                                 # n=520
        'Mixed': 'Mixed',                                   # n=290
        'not known': 'DNK',                                 # n=4
        'Other': 'Other',                                   # n=970
        'White': 'White',                                   # n=52689
        '': 'Blank',                                        # n=1
    },
    'excluded non white Eur': {
        'no': 'No',     # n=55530
        'yes': 'Yes',   # n=2372
        '': 'Blank',    # n=1
    },
    'StatinsEverYN': {
        ' ': 'Blank',   # n=6
        'DNK': 'DNK',   # n=22306
        'No': 'No',     # n=27448
        'Yes': 'Yes',   # n=8142
        '': 'Blank',    # n=1
    },
    'premature death': {
        'yes': 'Yes',   # n=1972
        '': 'Blank',    # n=
    },
    'eligible interval study': {
        'no': 'No',     # n=3726
        'yes': 'Yes',   # n=54176
        '': 'Blank',    # n=1
    },
    'detection': {
        'incident': 'incident',     # n=547
        'interval': 'interval',     # n=630
        'prevalent': 'prevalent',   # n=458
        '': 'Blank',                # n=56268
    },
    'presumed postmen BC': {
        'no': 'No',     # n=314
        'yes': 'Yes',   # n=1760
        '': 'Blank',    # n=55829
    },
    'Invasive or CIS or both': {
        'Both': 'Both',                                                                             # n=1143
        'Both?': 'Both',                                                                            # n=1
        'CIS': 'CIS',                                                                               # n=347
        'Definate cancer confirmed by MR 21/01/15. Op 15/01/15 awaiting pathology report.': 'DNK',  # n=1
        'Invasive': 'Invasive',                                                                     # n=517
        'Invasive ': 'Invasive',                                                                    # n=1
        '': 'Blank',                                                                                # n=55893
    },
    'invasive': {
        'no': 'No',     # n=347
        'yes': 'Yes',   # n=1663
        '': 'Blank',    # n=55893
    },
    'path': {
        'DCIS': 'DCIS',                                                 # n=6
        'DCIS & LCIS': 'DCIS',                                          # n=1
        'Encysted papillary carcinoma': 'EPC',                          # n=1
        'IDC': 'IDC',                                                   # n=18
        'IDC with DCIS': 'IDC with DCIS',                               # n=34
        'IL+EZ19:FB89C with LCIS': 'ILC with LCIS',                     # n=1
        'ILC': 'ILC',                                                   # n=1
        'ILC with LCIS': 'ILC with LCIS',                               # n=3
        'Invasive fibromatosis-like metaplastic carcinoma ': 'IMC',     # n=1
        'Invasive Lobular Carcinoma': 'ILC',                            # n=1
        'Invasive Lobular Carcinoma with DCIS and LCIS': 'ILC',         # n=1
        'Invasive Lobular Carcinoma with LCIS': 'ILC',                  # n=1
        'Invasive metaplastic carcinoma with DCIS': 'IMC',              # n=1
        'Invasive mucinous carcinom': 'IMC',                            # n=2
        'Invasive Mucinous Carcinoma with DCIS': 'IMC',                 # n=2
        'Invasive Tubular Carcinoma': 'ITC',                            # n=1
        'Invasive tubular carcinoma with DCIS': 'ITC',                  # n=1
        'metastatic': 'metastatic',                                     # n=1
        'Right: IDC with DCIS\n'
        'Left: IDC with DCIS': 'IDC with DCIS',   # n=1
        '': 'Blank',                                                    # n=57825
    },
    'LN': {
        '1': '1',                           # n=3
        '2': '2',                           # n=11
        '3': '3',                           # n=16
        '2 (provisional)': '2',             # n=2
        '3 (provisional)': '3',             # n=2
        'High': '3',                        # n=3
        'Right: 2\n'
        'Left: 1': '2',           # n=1
        'Tumour 1: 2\n'
        'Tumour 2: 2': '2',    # n=1
        '': 'Blank',                        # n=57864
    },
    'ER status': {
        'Negative': 'Negative',                                             # n=238
        'Negative - 0': 'Negative',                                         # n=8
        'Positive': 'Positive',                                             # n=1706
        'Positive ': 'Positive',                                            # n=3
        'Positive                                   Positive': 'Positive',  # n=1
        'Positive - 2': 'Positive',                                         # n=2
        'Positive - 4': 'Positive',                                         # n=1
        'Positive - 6': 'Positive',                                         # n=2
        'Positive - 7': 'Positive',                                         # n=2
        'Positive - 8': 'Positive',                                         # n=22
        'Positive, Positive': 'Positive',                                   # n=3
        'Positive`': 'Positive',                                            # n=1
        'Positve': 'Positive',                                              # n=1
        'Postive': 'Positive',                                              # n=3
        'Potitive': 'Positive',                                             # n=1
        'Right: Positive - 8\n'
        'Left: Positive - 8': 'Positive',                                   # n=1
        'Tumour 1: Positive - 8\n'
        'Tumour 2: Positive - 8': 'Positive',                               # n=1
        'Tumour 1: Positive - 8\n'
        'Tumour 2: Positive - 8\n'
        'Tumour 3: Positive - 8\n'
        'Tumour 4: Positive - 8\n'
        'Tumour 5: Positive - 8\n'
        'Tumour 6: Positive - 8': 'Positive',                               # n=1
        'unknown': 'DNK',                                                   # n=60
        '': 'Blank',                                                        # n=55846
    },
    'PR status': {
        'IDC Negative & ILC Positive ': 'Positive',                         # n=1
        'Negative': 'Negative',                                             # n=446
        'Negative - 0': 'Negative',                                         # n=14
        'Positive': 'Positive',                                             # n=1494
        'Positive ': 'Positive',                                            # n=7
        'Positive                                   Positive': 'Positive',  # n=1
        'Positive - 2': 'Positive',                                         # n=2
        'Positive - 3': 'Positive',                                         # n=1
        'Positive - 4': 'Positive',                                         # n=3
        'Positive - 5': 'Positive',                                         # n=1
        'Positive - 6': 'Positive',                                         # n=4
        'Positive - 7': 'Positive',                                         # n=6
        'Positive - 8': 'Positive',                                         # n=6
        'Postive': 'Positive',                                              # n=3
        'Right: Positive - 8\n'
        'Left: Positive - 8': 'Positive',                                   # n=1
        'Tumour 1: Positive - 6\n'
        'Tumour 2: Positive - 6': 'Positive',                               # n=1
        'Tumour 1: Positive - 8\n'
        'Tumour 2: Positive - 8\n'
        'Tumour 3: Positive - 8\n'
        'Tumour 4: Positive - 8\n'
        'Tumour 5: Positive - 8\n'
        'Tumour 6: Positive - 8': 'Positive',                               # n=1
        '': 'Blank',                                                        # n=55911
    },
    'HER2 status': {
        'n/a': 'n/a',                                                                           # n=9
        'N/A                                                             Negative': 'Negative', # n=1
        'Nagative': 'Negative',                                                                 # n=1
        'Negative - 0': 'Negative',                                                             # n=1453
        'Negative': 'Negative',                                                                 # n=13
        'Negative - 1+': 'Negative',                                                            # n=6
        'Negative - 2+ non amplified': 'Negative',                                              # n=15
        'Negative; Negative': 'Negative',                                                       # n=7
        'Not performed': 'Not performed',                                                       # n=38
        'Not reported': 'Not reported',                                                         # n=177
        'Positive': 'Positive',                                                                 # n=1
        'Positive ': 'Positive',                                                                # n=6
        'Positive - 3+': 'Positive',                                                            # n=1
        'Positive & Negative': 'Positive',                                                      # n=1
        'Positive (IDC)': 'Positive',                                                           # n=1
        'Positve': 'Positive',                                                                  # n=1
        'Postitive': 'Positive',                                                                # n=1
        'Postive': 'Positive',                                                                  # n=1
        'Right: Negative - 1+\n'
        'Left: Negative - 2+ non amplified': 'Negative',                                        # n=1
        'Tumour 1: Negative 1+\n'
        'Tumour 2: Negative 1+\n'
        'Tumour 3: Negative 1+\n'
        'Tumour 4: Negative 1+\n'
        'Tumour 5: Negative 1+\n'
        'Tumour 6: Negative 1+': 'Negative',                                                    # n=1
        'Tumour 1: Positive - 1+\n'
        'Tumour 2: Positive - 2+ non amplified': 'Positive',                                    # n=1
        '': 'Blank',                                                                            # n=56160
    },
    'SNPFHBC': {
        'yes': 'Yes',   # n=59
        '': 'Blank',    # n=57844
    },
    'Low density': {
        'no': 'No',     # n=49081
        'yes': 'Yes',   # n=8821
        '': 'Blank',    # n=1
    },
    'High density VaS': {
        'no': 'No',     # n=50327
        'yes': 'Yes',   # n=6671
        '': 'Blank',    # n=905
    },
    'PreviousCancerDiagnosis': {
        'No': 'No',     # n=56998
        'Yes': 'Yes',   # n=905
    },
    'Volpara done': {
        'yes': 'Yes',   # n=44965
        '': 'No',       # n=12938
    },
    'BiRads4': {
        'yes': 'Yes',   # n=529
        '': 'Blank',    # n=57374
    },
    'early menopause': {
        'yes': 'Yes',   # n=4268
        '': 'Blank',    # n=53635
    },


    'smol': {
        'no': 'No',     # n=
        'yes': 'Yes',   # n=
        '': 'Blank',    # n=
    },
    'bug': {
        'No': 'No',     # n=
        'Yes': 'Yes',   # n=
        '': 'Blank',    # n=
    },
}

numerical_columns = {  # r=range, b=number of blank, t=[non-numerical entries]
    'Age FDR OC': int,                  # r=(12-94),        b=56771, t=['Data not known']
    'age first prescribed': float,      # r=(39-75),        b=57587, t=[]
    'BMI 20': float,                    # r=(8 -94),        b= 9183, t=[]
    'BMI 202': float,                   # r=(8 -66),        b= 7260, t=[0, inaccurate, #REF!]
    'BMI20grp': int,                    # r=(1 - 4),        b= 9183, t=[]
    'BMI20grp2': float,                 # r=(1 - 4),        b= 7718, t=[inaccurate]
    'BMI20Grp3': float,                 # r=(1 -12),        b= 9185, t=[]
    'Time to DNA': float,               # r=(0 - 4),        b=48724, t=[]
    'HysterectomyAge': int,             # r=(18-72),        b=44810, t=['Data not known']
    'OvarianCancerAge': int,            # r=(23-75),        b=57834, t=[-99]
    'OvarianCancerAge2': int,           # r=(23-75),        b=57832, t=[-99]
    'AgeAtMenarche': int,               # r=(5 -75),        b=   15, t=[-99, 0]
    'AgeAtFirstPregnancy': int,         # r=(14-49),        b=    9, t=[-99, 0]
    'FFTP to age consent': float,       # r=(1 -39),        b=55717, t=[]
    'agefftp grp': int,                 # r=(0 - 5),        b=    1, t=[-99]
    'ChildrenNum': int,                 # r=(0 -15),        b=    1, t=[-99]
    'paritygrp': int,                   # r=(0 - 4),        b=    1, t=[]
    'OC\'EC from entry': float,         # r=(-32-7),        b=57738, t=[]
    'mss fh': int,                      # r=(0 -47),        b=57057, t=[death, FH, new]
    'Density Residual': float,          # r=(0 - 4),        b= 1147, t=[]
    'InitialTyrerCuzick': float,        # r=(1 -26),        b=    1, t=[]
    'TC8 only': float,                  # r=(1 -20),        b=    1, t=[]
    'TC8no wt': float,                  # r=(1 -23),        b=    1, t=[]
    'v8DR': float,                      # r=(0 -28),        b=    1, t=[]
    '10 yr avg': float,                 # r=(0 -30),        b=    1, t=[]
    'TC8 grp': int,                     # r=(1 - 6),        b=    1, t=[]
    'TCDR': float,                      # r=(0 -39),        b=    1, t=[#VALUE!, #N/A]
    'TCDRgrp': int,                     # r=(1 - 5),        b= 1173, t=[]
    'TC8DRgrp2': int,                   # r=(1 - 6),        b=    2, t=[]
    'DR': float,                        # r=(0 - 3),        b=    1, t=[]
    'TC8VpDR 10 yr avg': float,         # r=(0 -30),        b=    1, t=[]
    'DR Volpara': float,                # r=(0 - 3),        b=    1, t=[]
    'BMI': float,                       # r=(8 -75),        b=    3, t=[0]
    'BMI grp': int,                     # r=(1 - 4),        b= 3939, t=[]
    'AgeAtConsent': float,              # r=(46-84),        b=    1, t=[]
    'age grp': int,                     # r=(1 - 4),        b=    1, t=[]
    'age grp60': int,                   # r=(1 - 2),        b=    1, t=[]
    'time from 20': float,              # r=(26-64),        b=    1, t=[]
    'Age first mammo': float,           # r=(46-84),        b=    1, t=[]
    'Height_ft': int,                   # r=(4 - 8),        b=    6, t=[-99, 0]
    'Height_in': int,                   # r=(0 -12),        b=    6, t=[-99, 13]
    'Heightm': float,                   # r=(1 - 3),        b=    1, t=[-99, 0]
    'Height group': int,                # r=(1 - 3),        b= 1166, t=[]
    'Weight_st': int,                   # r=(5 -28),        b=    9, t=[-99, 0]
    'Weight_lb': int,                   # r=(5 -28),        b=    9, t=[-99, 0?]
    'WeightKg': float,                  # r=(35-202),       b=    3, t=[-99, 0]
    'WeightAt20_st': int,               # r=(4 -59),        b= 7832, t=[Data not known]
    'WeightAt20_lb': int,               # r=(0 -110),       b=18716, t=[Data not known]
    'WeightAt20_kg': int,               # r=(25-375),       b= 6285, t=[Data not known]
    'Wtkg20from stlb': float,           # r=(25-375),       b= 7836, t=[0]
    'ExerciseHoursPerMonth': int,       # r=(0 -120),       b=15658, t=[Data not known]
    'ExerciseMinsPerMonth': int,        # r=(0 -130),       b=29649, t=[Data not known]
    'Exercise Grp': int,                # r=(0 - 4),        b=    1, t=[99]
    'AlcoholUnitsPerWeek': int,         # r=(0 -1750),      b=18324, t=[Data not known]
    'alc grp': int,                     # r=(0 - 5),        b=  853, t=[Data not known]
    'alc grp2': int,                    # r=(0 - 5),        b=  853, t=[]
    'Wt gain': float,                   # r=(-4- 2),        b= 7865, t=[]
    'Wtgaingrp': int,                   # r=(1 - 6),        b= 7865, t=[]
    'Wtgaingrp2': int,                  # r=(1 - 6),        b= 7865, t=[]
    'wt gain per year': float,          # r=(-0- 0),        b= 7865, t=[#VALUE!]
    'OnHRTYears': int,                  # r=(0 -55),        b=  189, t=[-99, 0?]
    'OnHRTMonths': int,                 # r=(0 -60),        b=    7, t=[-99, 0?]
    'age at HRT': float,                # r=(7 -69),        b=47098, t=[]
    'HRT 10+post 50': float,            # r=(-3-10),        b=39855, t=[before, few, inter, yes]
    'HRT Last Used (Years)': int,       # r=(0 -70),        b=  108, t=[-99, 0?]
    'HRT Last Used (Months)': int,      # r=(0 -18),        b=   39, t=[-99, 0?]
    'StatinsYears': int,                # r=(0 -65),        b=52539, t=[Data not known]
    'StatinsMonths': int,               # r=(0 -30),        b=54351, t=[Data not known]
    'statins grp': int,                 # r=(0 - 3),        b=    1, t=[9, 0?]
    'age at death': float,              # r=(48-88),        b=55107, t=[]
    'age at censor to 70': float,       # r=(46-72),        b=    1, t=[]
    'follow up to 70': float,           # r=(-14-13),       b=    1, t=[]
    'Expected TC8nowt': float,          # r=(0-0.16),       b=    1, t=[]
    'fu to death': float,               # r=(0 -12),        b=    1, t=[]
    'Years of follow up': float,        # r=(0 -13),        b=    1, t=[]
    'Expected TC8DR': float,            # r=(0-0.33),       b=    1, t=[]
    'expected TC8': float,              # r=(0-0.21),       b=    1, t=[]
    'Expected TC6': float,              # r=(0-0.23),       b=    1, t=[]
    'age bc': float,                    # r=(46-83),        b=55754, t=[]
    'age bc grp': int,                  # r=(1 - 3),        b=56422, t=[]
    'age BCgrp2': int,                  # r=(1 - 4),        b=55831, t=[]
    'size': float,                      # r=(3-120),        b=57903, t=[it's all in mm and some have multiple tumours]
    'grade invasive': int,              # r=(0 -32),        b=56157, t=[lots of text entry and maybe grade/breast]
    'ER score': int,                    # r=(0 - 8),        b=55797, t=[lots of text entry and maybe score/breast]
    'PR score': int,                    # r=(0 - 8),        b=55797, t=[lots of text entry and maybe score/breast]
    'MSS family': int,                  # r=(0 - 47),       b=56575, t=[]
    'Manchester score proband': int,    # r=(2 - 6),        b=56377, t=[]
    'path grade': int,                  # r=(-2 - 2),       b=56697, t=[]
    'Hormonal ER': int,                 # r=(-3 - 4),       b=56445, t=[]
    'HER2': int,                        # r=(-6 - 3),       b=56726, t=[]
    'MSS personal': int,                # r=(-7 -12),       b=56377, t=[]
    'prev BC score': int,               # r=(2 -13),        b=56997, t=[]
    'total MSS': int,                   # r=(-5 -48),       b=56379, t=[]
    'VASCombinedAvDensity': float,      # r=(1 -97),        b= 6418, t=[-99, -43.75]
    'age previous bc': float,           # r=(24-77),        b=56999, t=[-43.1594798083504]
    'ageprBC grp': int,                 # r=(1 - 5),        b=56999, t=[]
    'VBD%': float,                      # r=(1 -35),        b=12939, t=[]
    'FGV cm3': float,                   # r=(4-349),        b=12939, t=[]
    'age menopauuse': int,              # r=(20-65),        b=   49, t=[-99, 0]
}

function_mapping = {
    'ProcID': too_many_discrete,                        # unique names for all patients
    'ProcID2': too_many_discrete,                       # (a copy of?) unique names for all patients
    'VAS.ProcID': too_many_discrete,                    # (a copy of a copy??) unique names for all patients
    'volpara.ProcID': too_many_discrete,                # (a copy of a copy of a copy???) unique names for all patients
    'possible missed cancer not in main data': too_many_discrete,  # some names and random entries
    'dod': date_to_number,                              # r=(07/05/2010-18/01/2022), b=57903, t=[]
    'Prescribed Chemoprevention Date': date_to_number,  # r=(01/06/1996-30/12/2019), b=57587, t=[]
    'Date of Entry FHC': date_to_number,                # r=(27/04/1987-28/01/2020), b=56547, t=[]
    'contralateral date': date_to_number,               # r=(30/04/2010-17/07/2020), b=57801, t=[b]
    'DateOfDeath': date_to_number,                      # r=(03/04/2010-03/09/2020), b=55547, t=[]
    # an updated version of the previous column?
    'DateOfDeath2': date_to_number,                     # r=(03/04/2010-18/01/2022), b=54849, t=[]
    'DateOfDeath3': date_to_number,                     # r=(03/04/2010-18/01/2022), b=54868, t=[]
    'DateOfCancerDiagnosis': date_to_number,            # r=(08/01/2011-03/09/2021), b=55772, t=[only months or years]
    # almost a duplicate of previous column but dates are slightly different
    'DateOfCancerDiagnosis2': date_to_number,           # r=(15/11/2009-03/09/2021), b=55839, t=[only months or years]
    'DateOfCancerDiagnosis3': date_to_number,           # r=(15/11/2009-03/09/2021), b=55747, t=[]
    'AddressLine1': address_to_gps,                     #                            b=    4
    'DC 1a': too_many_discrete,                         # r=(108 others, 61 yes),    b=57669, t=[cause of deaths?]
    'Location': too_many_discrete,                      # r=(1 not known, 3956 Loc), b=47872, t=[39 different locations]
    'date ec/oc': date_to_number,                       # r=(05/03/2012-12/10/2016), b=57758, t=[]
    'Mammo1-Date': date_to_number,                      # r=(26/05/2009-25/06/2015), b=    1, t=[]
    'DOB': date_to_number,                              # r=(06/06/1926-29/12/1968), b=    1, t=[]
    'Date last follow up': date_to_number,              # r=(15/11/2009-18/01/2022), b=    1, t=[]
    'Date last follow or death': date_to_number,        # r=(03/04/2010-18/01/2022), b=    1, t=[]
    'DateOfPreviousDiagnosis': date_to_number,          # r=(01/11/1975-24/04/2013), b=56999, t=[]
    'HRTName': too_many_discrete,                       # r=(1 HRT, 2427 Data not known), b=48307, t=[666 different HRT]
}


if not csv_processed_previously:
    proCeSsVed = pd.DataFrame(columns=csv_data.columns)
    
    # map csv entries to pre-specified entries
    for col in proCeSsVed.columns:
        print("Currently processing column:", col)
        if col in function_mapping:
            proCeSsVed[col] = csv_data[col].apply(function_mapping[col])
        elif col in string_mapping:
            proCeSsVed[col] = csv_data[col].fillna("").map(string_mapping[col]).fillna(csv_data[col])
        elif col in numerical_columns:
            proCeSsVed[col] = csv_data[col]
        else:
            print(f"\n\t Col:{col} does not exist in mapping\n")
            proCeSsVed[col] = csv_data[col]

    # Create new columns
    proCeSsVed["Subtype"] = proCeSsVed.apply(
        lambda r:
        "HER2-enriched" if r["HER2 status"] == "Positive" and
                           r["ER status"] == "Negative" and
                           r["PR status"] == "Negative" else
        "Triple-negative" if r["HER2 status"] == "Negative" and
                             r["ER status"] == "Negative" and
                             r["PR status"] == "Negative" else
        "Luminal A (HER2-)" if r["HER2 status"] == "Negative" and
                               (r["ER status"] == "Positive" or r["PR status"] == "Positive") else
        "Luminal B (HER2+)" if r["HER2 status"] == "Positive" and
                               (r["ER status"] == "Positive" or r["PR status"] == "Positive") else
        "Blank",
        axis=1
    )

    proCeSsVed['ER_PR_HER2'] = proCeSsVed.apply(
        lambda r:
        "E+P+H+" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "E+P+H-" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "E+P-H+" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "E+P-H-" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "E-P+H+" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "E-P+H-" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "E-P-H+" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "E-P-H-" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "Blank",
        axis=1
    )

    # space to add VAS readings from MAI-VAS and MADAM

    # space to add other csvs which contain things like Ki-67

    # Save to new CSV
    proCeSsVed.to_csv(os.path.join(save_dir, save_name), index=False)
else:
    proCeSsVed = pd.read_csv(os.path.join(save_dir, save_name), sep=',')

'''
Data types in the csv:
    - Categorical (strings)
        in string_mapping
    - Ordinal (integer numbers)
        in numerical_columns = int
    - Continuous (not integer numbers)
        in numerical_columns = float
    - Dates (Converted to ordinal but converted back to display)
        in function_mapping = date_to_number
    - Location (Converted postcode to coords
        in function_mapping = address_to_gps  
'''

def get_type(col):
    if col in string_mapping:
        return "CAT"
    if col in function_mapping:
        if function_mapping[col] == date_to_number:
            return "DATE"
        if function_mapping[col] == address_to_gps:
            return "LOC"
    if col in numerical_columns:
        if numerical_columns[col] == int:
            return "ORD"
        else:
            return "CONT"
    return "CAT"

def chi2_and_cramers_v(x, y):
    table = pd.crosstab(x, y)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return {}
    chi2, p, dof, exp = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    return {"Chi²": chi2, "df": dof, "p": p, "Cramér's V": v}

def pearson_spearman(x, y):
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {}
    res = {}
    r, p = stats.pearsonr(x, y)
    res["Pearson r"] = r
    res["Pearson p"] = p
    rho, sp = stats.spearmanr(x, y)
    res["Spearman ρ"] = rho
    res["Spearman p"] = sp
    return res

def anova_cat_cont(cat, cont):
    df = pd.DataFrame({"cat": cat, "cont": cont}).dropna()
    groups = [g["cont"].values for _, g in df.groupby("cat")]
    if len(groups) < 2:
        return {}
    f, p = stats.f_oneway(*groups)
    return {"ANOVA F": f, "ANOVA p": p}

class DataExplorer(tk.Tk):
    def __init__(self, df):
        super().__init__()
        self.df = df

        self.title("Interactive Data Explorer")
        self.geometry("1100x700")

        cols = list(df.columns)

        # Choose sensible defaults
        default_x = "DOB" if "DOB" in cols else cols[0]
        if "Subtype" in cols and "Subtype" != default_x:
            default_y = "Subtype"
        else:
            default_y = cols[1] if len(cols) > 1 else cols[0]

        # --- Top control panel ------------------------------------------------
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack(side="top", fill="x")

        ttk.Label(ctrl, text="X column:").pack(side="left", padx=(0, 5))
        self.x_var = tk.StringVar(value=default_x)
        self.x_box = ttk.Combobox(
            ctrl, textvariable=self.x_var, values=cols, state="readonly", width=30
        )
        self.x_box.pack(side="left", padx=(0, 20))

        ttk.Label(ctrl, text="Y column:").pack(side="left", padx=(0, 5))
        self.y_var = tk.StringVar(value=default_y)
        self.y_box = ttk.Combobox(
            ctrl, textvariable=self.y_var, values=cols, state="readonly", width=30
        )
        self.y_box.pack(side="left", padx=(0, 20))

        # Swap button
        self.swap_btn = ttk.Button(ctrl, text="Swap X/Y", command=self.swap_axes)
        self.swap_btn.pack(side="left", padx=(0, 10))

        self.update_btn = ttk.Button(ctrl, text="Update plot", command=self.update_plot)
        self.update_btn.pack(side="left", padx=(0, 20))

        # --- Main layout: plot (left) + stats (right) ------------------------
        main = ttk.Frame(self, padding=10)
        main.pack(side="top", fill="both", expand=True)

        # Matplotlib figure
        fig_frame = ttk.Frame(main)
        fig_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.cbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = ttk.Frame(main)
        stats_frame.pack(side="right", fill="y")

        ttk.Label(
            stats_frame, text="Statistical tests", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w")
        self.stats_text = tk.Text(stats_frame, width=35, height=35)
        self.stats_text.pack(fill="both", expand=True, pady=(5, 0))

        # checkbuttons for excluding "Blank"
        self.x_exclude_blank = tk.BooleanVar(value=False)
        self.y_exclude_blank = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            ctrl,
            text="Exclude 'Blank' in X",
            variable=self.x_exclude_blank,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Checkbutton(
            ctrl,
            text="Exclude 'Blank' in Y",
            variable=self.y_exclude_blank,
            command=self.update_plot,
        ).pack(side="left")

        # Initial plot
        self.update_plot()

    def swap_axes(self):
        """Swap selected X and Y columns."""
        x = self.x_var.get()
        y = self.y_var.get()
        self.x_var.set(y)
        self.y_var.set(x)
        self.update_plot()

    def update_plot(self):
        x_col = self.x_var.get()
        y_col = self.y_var.get()

        x_full = self.df[x_col]
        y_full = self.df[y_col]

        # build a mask for excluding "Blank"
        mask = pd.Series(True, index=self.df.index)
        if self.x_exclude_blank.get():
            mask &= (x_full != "Blank")
        if self.y_exclude_blank.get():
            mask &= (y_full != "Blank")

        x = x_full[mask]
        y = y_full[mask]

        tx, ty = get_type(x_col), get_type(y_col)

        # --- FULL RESET OF FIGURE & COLORBAR -----------------------------------
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if hasattr(self, "cbar") and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # --- PLOT ------------------------------------------------------------
        # 1) Both numeric-ish (CONT/ORD/DATE)
        if tx in ["CONT", "ORD", "DATE"] and ty in ["CONT", "ORD", "DATE"]:
            self.ax.scatter(x, y, alpha=0.6)
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel(y_col)

            # DATE formatting: here DATE can be on X and/or Y
            if tx == "DATE":
                xticks = self.ax.get_xticks()
                self.ax.set_xticks(xticks)
                self.ax.set_xticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in xticks
                    ],
                    rotation=45,
                    ha="right",
                )
            if ty == "DATE":
                yticks = self.ax.get_yticks()
                self.ax.set_yticks(yticks)
                self.ax.set_yticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in yticks
                    ]
                )

        # 2) CAT (X) vs numeric/DATE (Y): categories on X, numeric on Y
        elif tx == "CAT" and ty in ["CONT", "ORD", "DATE"]:
            df_plot = pd.DataFrame({"cat": x.astype(str), "val": y})
            df_plot.boxplot(column="val", by="cat", ax=self.ax)
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel(y_col)
            self.ax.set_title("")
            self.fig.suptitle("")

            # DATE on Y-axis only (because X is CAT)
            if ty == "DATE":
                yticks = self.ax.get_yticks()
                self.ax.set_yticks(yticks)
                self.ax.set_yticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in yticks
                    ]
                )

        # 3) numeric/DATE (X) vs CAT (Y): numeric on X, categories on Y
        elif ty == "CAT" and tx in ["CONT", "ORD", "DATE"]:
            df_plot = pd.DataFrame({"cat": y.astype(str), "val": x})
            df_plot.boxplot(column="val", by="cat", ax=self.ax)
            self.ax.set_xlabel(y_col)
            self.ax.set_ylabel(x_col)
            self.ax.set_title("")
            self.fig.suptitle("")

            # IMPORTANT FIX: DATE is on Y-axis (val), not X
            if tx == "DATE":
                yticks = self.ax.get_yticks()
                self.ax.set_yticks(yticks)
                self.ax.set_yticklabels(
                    [number_to_date(int(round(t))) if not np.isnan(t) else "" for t in yticks]
                )

        # 4) CAT–CAT heatmap
        elif tx == "CAT" and ty == "CAT":
            table = pd.crosstab(x, y)
            if not table.empty:
                im = self.ax.imshow(table, aspect="auto")
                self.ax.set_xticks(range(len(table.columns)))
                self.ax.set_xticklabels(table.columns, rotation=90)
                self.ax.set_yticks(range(len(table.index)))
                self.ax.set_yticklabels(table.index)
                self.ax.set_xlabel(y_col)
                self.ax.set_ylabel(x_col)
                self.ax.set_title("Count heatmap")
                self.cbar = self.fig.colorbar(
                    im, ax=self.ax, fraction=0.046, pad=0.04
                )
            else:
                self.ax.text(0.5, 0.5, "No data", ha="center", va="center")
                self.ax.set_axis_off()

        # 5) Fallback: anything else → scatter
        else:
            self.ax.scatter(x, y, alpha=0.6)
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel(y_col)

            # If one side is DATE here, treat like case (1)
            if tx == "DATE":
                xticks = self.ax.get_xticks()
                self.ax.set_xticks(xticks)
                self.ax.set_xticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in xticks
                    ],
                    rotation=45,
                    ha="right",
                )
            if ty == "DATE":
                yticks = self.ax.get_yticks()
                self.ax.set_yticks(yticks)
                self.ax.set_yticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in yticks
                    ]
                )

        self.fig.tight_layout()
        self.canvas.draw()

        # --- Stats -----------------------------------------------------------
        stats_results = {}
        pair = (tx, ty)

        # CAT–CAT
        if tx == "CAT" and ty == "CAT":
            stats_results.update(chi2_and_cramers_v(x, y))

        # CONT/ORD/DATE vs CONT/ORD/DATE
        if tx in ["CONT", "ORD", "DATE"] and ty in ["CONT", "ORD", "DATE"]:
            stats_results.update(
                pearson_spearman(
                    pd.to_numeric(x, errors="coerce"),
                    pd.to_numeric(y, errors="coerce"),
                )
            )

        # CAT vs CONT-ish → ANOVA
        if tx == "CAT" and ty in ["CONT", "ORD", "DATE"]:
            stats_results.update(anova_cat_cont(x, pd.to_numeric(y, errors="coerce")))
        if ty == "CAT" and tx in ["CONT", "ORD", "DATE"]:
            stats_results.update(anova_cat_cont(y, pd.to_numeric(x, errors="coerce")))

        # LOC: for now treat as continuous (you can swap in Moran / Mantel later)
        if "LOC" in pair and ("CONT" in pair or "ORD" in pair or "DATE" in pair):
            stats_results.update(
                pearson_spearman(
                    pd.to_numeric(x, errors="coerce"),
                    pd.to_numeric(y, errors="coerce"),
                )
            )

        # Show stats
        self.stats_text.delete("1.0", tk.END)
        if stats_results:
            for k, v in stats_results.items():
                if isinstance(v, (int, float, np.floating)):
                    self.stats_text.insert(tk.END, f"{k}: {v:.4g}\n")
                else:
                    self.stats_text.insert(tk.END, f"{k}: {v}\n")
        else:
            self.stats_text.insert("1.0", "No implemented tests for this combination yet.")

# -----------------------------------------------------------------------------
# 4. RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    matplotlib.use("TkAgg")  # optional, usually default
    app = DataExplorer(proCeSsVed)
    app.mainloop()

print("Done")
