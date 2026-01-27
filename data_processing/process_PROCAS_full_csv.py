import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys
import re
import datetime as dt
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    - In normal Python: folder containing the script / exe
    - In PyInstaller .exe: the temporary _MEIPASS folder
    """
    if hasattr(sys, "_MEIPASS"):  # PyInstaller bundled
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, relative_path)

csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
save_dir = csv_directory  # 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/processed_data'

csv_name = 'PROCAS_full_data_16-03-2024.csv'
csv_file_pointer = os.path.join(csv_directory, csv_name)
cancers_name = 'PROCAS_CANCER_DATABASE_first_process.csv'
cancers_file_pointer = os.path.join(csv_directory, cancers_name)
save_name = 'processed_PROCAS_full_data_with_cancer_data.csv'

csv_data = pd.read_csv(csv_file_pointer, sep=',')
cancers_data = pd.read_csv(cancers_file_pointer, sep=',')

csv_processed_previously = False

class CaseInsensitiveDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def _k(self, key):
        return key.casefold() if isinstance(key, str) else key

    def __setitem__(self, key, value):
        super().__setitem__(self._k(key), value)

    def __getitem__(self, key):
        return super().__getitem__(self._k(key))

    def get(self, key, default=None):
        return super().get(self._k(key), default)

    def __contains__(self, key):
        return super().__contains__(self._k(key))

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

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

def broken_dates(n):
    if n is None:
        return ""
    try:
        d = dt.date.fromordinal(int(n))
        return d.strftime("%d/%m/%Y")
    except Exception:
        return ""

def extract_subtypes_from_path(column):
    print("extracting PATH columns")
    column_indexes = {
        'DCIS': 0,
        'IDC': 1,
        'LCIS': 2,
        'Metastatic': 3,
        'Mucinous': 4,
        'Phyllodes': 5,
        'Papillary': 6,
        'Apocrine': 7,
        'Adenoid Cystic': 8,
        'Metaplastic': 9,
        'Medullary': 10,
        'Tubular': 11,
        'ILC': 12,
        'Invasive Cribriform': 13,
        'DNK': 14,
    }
    path_entries = CaseInsensitiveDict({
        ' ': ['DNK'],                                                                                           # n=1
        'DCIS': ['DCIS'],                                                                                       # n=300
        'DCIS ': ['DCIS'],                                                                                      # n=2
        'DCIS & Hybrid DCIS/LCIS': ['DCIS', 'LCIS'],                                                            # n=1
        'DCIS (anterior: comedo necrosis / posterior: solid, cribriform & comedo type DCIS)': ['DCIS'],         # n=1
        'DCIS (CRIBIFORM) ': ['DCIS'],                                                                          # n=1
        'DCIS (cribriform & flat)': ['DCIS'],                                                                   # n=1
        'DCIS (cribriform & papillary with focal necrosis) & LCIS': ['DCIS', 'LCIS'],                            # n=1
        'DCIS (cribriform & solid)': ['DCIS'],                                                                  # n=1
        'DCIS (cribriform)': ['DCIS'],                                                                          # n=3
        'DCIS (micro-papillary & comedo)': ['DCIS'],                                                            # n=1
        'DCIS (micropapillary, cribriform & flat)': ['DCIS'],                                                   # n=1
        'DCIS (solid & comedo)': ['DCIS'],                                                                      # n=3
        'DCIS (solid & cribriform)': ['DCIS'],                                                                  # n=5
        'DCIS (solid, cribriform & micropapillary)': ['DCIS'],                                                  # n=1
        'DCIS (solid, papillary & comedo with areas of lobular cancerisation).': ['DCIS'],                      # n=1
        'DCIS + Invasive mucinous carcinoma': ['DCIS', 'Mucinous'],                                             # n=1
        'DCIS and LCIS': ['DCIS', 'LCIS'],                                                                      # n=5
        'DCIS Comedonecrosis': ['DCIS'],                                                                        # n=1
        'DCIS intermediate grade;solid cribriform': ['DCIS'],                                                   # n=1
        'DCIS solid': ['DCIS'],                                                                                 # n=1
        'DCIS solid & comedo': ['DCIS'],                                                                        # n=1
        'DCIS with IDC + LCIS': ['DCIS', 'IDC', 'LCIS'],                                                        # n=1
        'DCIS with lobular cancerisation & malignant calcification': ['DCIS', 'ILC'],                           # n=1
        'DCIS with microinvasion': ['DCIS'],                                                                    # n=2
        'DCIS/encysted papillary carcinoma': ['DCIS', 'Papillary'],                                             # n=1
        'Encapsulated papillary carcinoma with peripheral foci of invasive mucinous carcinoma with DCIS':
        ['Papillary', 'Mucinous', 'DCIS'],                                                                      # n=1
        'Encysted papillary carcinoma': ['Papillary'],                                                          # n=2
        'Encysted papillary carcinoma with DCIS': ['DCIS', 'Papillary'],                                        # n=2
        'EPC and DCIS': ['DCIS'],                                                                               # n=1
        'Foci of Microcalcification with DCIS': ['DCIS'],                                                       # n=1
        'High Grade Neuroendocrine Carcinoma/Small Cell Carcinoma with DCIS': ['DCIS'],                         # n=1
        'IDC': ['IDC'],                                                                                         # n=324
        'IDC ': ['IDC'],                                                                                        # n=11
        'IDC  ': ['IDC'],                                                                                       # n=2
        'IDC   ': ['IDC'],                                                                                      # n=1
        'IDC  (micro-papillary)': ['IDC', 'Papillary'],                                                         # n=1
        'IDC - Lobular (alveolar variant)': ['IDC', 'LCIS'],                                                    # n=1
        'IDC - solid papillary / ductal': ['IDC'],                                                              # n=1
        'IDC & ILC (mixed) with LCIS & DCIS (Cribriform & flat)': ['IDC', 'ILC', 'LCIS', 'DCIS'],               # n=1
        'IDC & Invasive Lobular carcinoma (Mixed) with LCIS': ['IDC', 'ILC', 'LCIS'],                           # n=1
        'IDC & Lobular Carcinoma': ['IDC', 'ILC'],                                                              # n=1
        'IDC (cribriform) with DCIS (cribriform)': ['IDC', 'DCIS'],                                             # n=1
        'IDC (localised)': ['IDC'],                                                                             # n=1
        'IDC (mucinous & papillary components) with DCIS': ['IDC', 'DCIS'],                                     # n=1
        'IDC (papillary) with DCIS': ['IDC', 'DCIS'],                                                           # n=1
        'IDC + DCIS': ['IDC', 'DCIS'],                                                                          # n=24
        'IDC + DCIS - cribriform': ['IDC', 'DCIS'],                                                             # n=1
        'IDC + DCIS - Intermediate/high grade': ['IDC', 'DCIS'],                                                # n=1
        'IDC + LCIS': ['IDC', 'LCIS'],                                                                          # n=1
        'IDC and ILC with DCIS and LCIS': ['IDC', 'ILC', 'DCIS', 'LCIS'],                                       # n=1
        'IDC or Invasive cribform': ['IDC', 'Invasive Cribriform'],                                             # n=1
        'IDC with apocrine features': ['IDC'],                                                                  # n=2
        'IDC with apocrine features & DCIS': ['IDC', 'DCIS'],                                                   # n=1
        'IDC with DCIS': ['IDC', 'DCIS'],                                                                       # n=865
        'IDC with DCIS ': ['IDC', 'DCIS'],                                                                      # n=6
        'IDC with DCIS & LCIS': ['IDC', 'DCIS', 'LCIS'],                                                        # n=1
        'IDC with DCIS (Apocrine type)': ['IDC', 'DCIS'],                                                       # n=1
        'IDC with DCIS (biopsy)    DCIS with microinvasion (surgery post chemo)': ['IDC', 'DCIS'],              # n=1
        'IDC with DCIS (cribrifirm & solid with focal comedo necrosis)': ['IDC', 'DCIS'],                       # n=1
        'IDC with DCIS (cribriform)': ['IDC', 'DCIS'],                                                          # n=7
        'IDC with DCIS (cribriform) Tumour 1': ['IDC', 'DCIS'],                                                 # n=1
        'IDC with DCIS (papillary)': ['IDC', 'DCIS'],                                                           # n=1
        'IDC with DCIS (papillary, flat, solid, cribriform)': ['IDC', 'DCIS'],                                  # n=1
        'IDC with DCIS (solid & comedo)': ['IDC', 'DCIS'],                                                      # n=2
        'IDC with DCIS (solid)': ['IDC', 'DCIS'],                                                               # n=2
        'IDC with DCIS (solid, cribriform & comedo-type)': ['IDC', 'DCIS'],                                     # n=1
        'IDC with DCIS and LCIS': ['IDC', 'DCIS', 'LCIS'],                                                      # n=9
        'IDC with DCIS, lobular features': ['IDC', 'DCIS'],                                                     # n=1
        'IDC with DCIS+LCIS': ['IDC', 'DCIS', 'LCIS'],                                                          # n=1
        'IDC with Encysted papillary carcinoma': ['IDC', 'Papillary'],                                          # n=1
        'IDC with hybrid DCIS/LCIS': ['IDC', 'DCIS', 'LCIS'],                                                   # n=1
        'IDC with LCIS': ['IDC', 'LCIS'],                                                                       # n=6
        'IDC with lobular features': ['IDC'],                                                                   # n=4
        'IDC with lobular features & DCIS': ['IDC', 'DCIS'],                                                    # n=1
        'IDC with Lobular features + DCIS': ['IDC', 'DCIS'],                                                    # n=3
        'IDC with Lobular features + DCIS & LCIS': ['IDC', 'DCIS', 'LCIS'],                                     # n=1
        'IDC with Lobular Features with DCIS': ['IDC', 'DCIS'],                                                 # n=6
        'IDC with lobular Neoplasia In Situ': ['IDC', 'LCIS'],                                                  # n=1
        'IDC with lobular pattern': ['IDC'],                                                                    # n=1
        'IDC with micropapillary growth & DCIS (solid)': ['IDC', 'Papillary', 'DCIS'],                          # n=1
        'IDC, NOS with DCIS': ['IDC', 'DCIS'],                                                                  # n=1
        'IDC+ DCIS': ['IDC', 'DCIS'],                                                                           # n=1
        'IDC+DCIS': ['IDC', 'DCIS'],                                                                            # n=15
        'IDC+DCIS ': ['IDC', 'DCIS'],                                                                           # n=1
        'IDC+DCIS - intermediate with solid and cribriform':['IDC', 'DCIS'],                                    # n=1
        'IDC+DCIS - two areas': ['IDC', 'DCIS'],                                                                # n=1
        'ILC with DCIS': ['ILC', 'DCIS'],                                                                       # n=1
        'ILC with LCIS': ['ILC', 'LCIS'],                                                                       # n=1
        'Infiltrating Duct Carcinoma': ['IDC'],                                                                 # n=4
        'Infiltrating Duct Carcinoma with DCIS': ['IDC', 'DCIS'],                                               # n=1
        'Infiltrating Duct Carcinoma, NOS': ['IDC'],                                                            # n=3
        'Infiltrating Duct mixed with other types of caricnoma': ['IDC'],                                       # n=1
        'Infiltrating Ductal Carcinoma with DCIS': ['IDC', 'DCIS'],                                             # n=1
        'Infiltrating Metastatic Carcinoma': ['Metastatic'],                                                    # n=1
        'Intracystic papillary carcinoma with DCIS': ['DCIS'],                                                  # n=1
        'Intracystic papillary carcinoma with DCIS ': ['DCIS'],                                                 # n=1
        'Intraductal carcinoma non-infiltrating': ['DCIS'],                                                     # n=1
        'Intraductal Carcinoma, non-infiltrating': ['DCIS'],                                                    # n=1
        'Invasive': ['Invasive Cribriform'],                                                                    # n=1
        'Invasive Adenoid Cystic Carcinoma with LCIS': ['Adenoid Cystic', 'LCIS'],                              # n=1
        'Invasive Apocine Carcinoma ': ['Apocrine'],                                                            # n=1
        'Invasive Apocrine Carcinoma': ['Apocrine'],                                                            # n=2
        'Invasive Apocrine Carcinoma with DCIS': ['Apocrine', 'DCIS'],                                          # n=3
        'Invasive apocrine carcinoma with DCIS ': ['Apocrine', 'DCIS'],                                         # n=1
        'Invasive Carcinoma': ['Invasive Cribriform'],                                                          # n=1
        'Invasive Carcinoma (micro-papillary)': ['Invasive Cribriform', 'Papillary'],                           # n=1
        'Invasive Carcinoma with apocrine features': ['Invasive Cribriform'],                                   # n=1
        'Invasive Carcinoma with Lobular Features': ['Invasive Cribriform'],                                    # n=1
        'Invasive Carcinoma with Medullary features': ['Invasive Cribriform'],                                  # n=1
        'Invasive cribiform carcinoma': ['Invasive Cribriform'],                                                # n=2
        'Invasive cribiform carcinoma with DCIS': ['Invasive Cribriform', 'DCIS'],                              # n=1
        'Invasive Cribriform Carcinoma, with DCIS, Lobular': ['Invasive Cribriform', 'DCIS'],                   # n=1
        'Invasive Ductal & Lobular Carcinoma with DCIS': ['IDC', 'ILC', 'DCIS'],                                # n=1
        'Invasive Ductal/Lobular Carcinoma with LCIS': ['IDC', 'ILC', 'LCIS'],                                  # n=2
        'Invasive Ductulo-Lobular Carcinoma': ['IDC', 'ILC'],                                                   # n=2
        'INVASIVE LOBULAR': ['ILC'],                                                                            # n=3
        'Invasive lobular carcinoma': ['ILC'],                                                                  # n=41
        'Invasive lobular carcinoma ': ['ILC'],                                                                 # n=1
        'Invasive Lobular Carcinoma with DCIS': ['ILC', 'DCIS'],                                                # n=6
        'Invasive lobular carcinoma with DCIS  (cribriform)            ': ['ILC', 'DCIS'],                      # n=1
        'Invasive Lobular Carcinoma with DCIS and LCIS': ['ILC', 'DCIS', 'LCIS'],                               # n=6
        'Invasive lobular carcinoma with LCIS': ['ILC', 'LCIS'],                                                # n=91
        'Invasive lobular carcinoma with LCIS  ': ['ILC', 'LCIS'],                                              # n=2
        'Invasive Lobular Carcnioma with LCIS': ['ILC', 'LCIS'],                                                # n=1
        'Invasive lobular with LCIS': ['ILC', 'LCIS'],                                                          # n=2
        'Invasive lobular with lobular neoplasia in situ': ['ILC', 'LCIS'],                                     # n=1
        'Invasive Lobular/Ductal Carcinoma': ['ILC', 'IDC'],                                                    # n=1
        'Invasive Lobular/Ductal Carcinoma with DCIS and LCIS': ['ILC', 'IDC', 'DCIS', 'LCIS'],                 # n=1
        'Invasive lobular/ductal carcinoma with LCIS': ['ILC', 'IDC', 'LCIS'],                                  # n=1
        'Invasive medullary-like carcinoma': ['Medullary'],                                                     # n=2
        'Invasive metaplastic carcinoma ': ['Metaplastic'],                                                     # n=1
        'Invasive metaplastic carcinoma with DCIS': ['Metaplastic', 'DCIS'],                                    # n=2
        'Invasive metaplastic/matrix-producing carcinoma': ['Metaplastic'],                                     # n=1
        'Invasive micropapillary + DCIS. ': ['Papillary', 'DCIS'],                                              # n=1
        'Invasive Micropapillary Carcinoma': ['Papillary'],                                                     # n=3
        'Invasive Micropapillary Carcinoma with DCIS': ['Papillary', 'DCIS'],                                   # n=3
        'Invasive Micropapillary Carcinoma with DCIS and LCIS': ['Papillary', 'DCIS', 'LCIS'],                  # n=1
        'Invasive mixed lobular-ductal carcinoma with DCIS and LCIS': ['ILC', 'IDC', 'DCIS', 'LCIS'],           # n=1
        'Invasive Mucinous Carcinoma': ['Mucinous'],                                                            # n=13
        'Invasive mucinous carcinoma ': ['Mucinous'],                                                           # n=2
        'Invasive Mucinous Carcinoma with DCIS': ['Mucinous', 'DCIS'],                                          # n=12
        'Invasive mucinous carcinoma with DCIS ': ['Mucinous', 'DCIS'],                                         # n=1
        'Invasive mucinous carinoma with DCIS': ['Mucinous', 'DCIS'],                                           # n=1
        'invasive mucinous with DCIS': ['Mucinous', 'DCIS'],                                                    # n=1
        'Invasive Papillary Carcinoma with DCIS': ['Papillary', 'DCIS'],                                        # n=1
        'Invasive Pleomorhism Lobular Carcinoma with LCIS': ['ILC', 'LCIS'],                                    # n=1
        'Invasive Pleomorphic Lobular/Ductal Carcinoma with LCIS & DCIS': ['ILC', 'IDC', 'LCIS'],               # n=1
        'Invasive Tubular Carcinoma': ['Tubular'],                                                              # n=10
        'Invasive tubular carcinoma ': ['Tubular'],                                                             # n=1
        'Invasive Tubular Carcinoma & DCIS': ['Tubular', 'DCIS'],                                               # n=1
        'Invasive tubular carcinoma with DCIS': ['Tubular', 'DCIS'],                                            # n=8
        'Invasive Tubular Carcinoma with DCIS & LCIS': ['Tubular', 'DCIS', 'LCIS'],                             # n=1
        'Invasive Tubular Carcinoma with DCIS and LCIS': ['Tubular', 'DCIS', 'LCIS'],                           # n=1
        'Invasive tubular carcinoma with LCIS': ['Tubular', 'LCIS'],                                            # n=1
        'Invasive Tubular Carcninoma with DCIS': ['Tubular', 'DCIS'],                                           # n=1
        'Invasive tubular mixed': ['Tubular'],                                                                  # n=1
        'Invasive tubular with DCIS': ['Tubular', 'DCIS'],                                                      # n=2
        'Invasive Tubulolobular': ['Tubular'],                                                                  # n=1
        'Invasive Tubulo-lobular Carcinoma with LCIS': ['Tubular', 'LCIS'],                                     # n=1
        'LCIS + DCIS': ['LCIS', 'DCIS'],                                                                        # n=1
        'LCIS with DCIS': ['LCIS', 'DCIS'],                                                                        # n=1
        'Lobular': ['ILC'],                                                                                     # n=1
        'Lobular Carcinoma': ['ILC'],                                                                           # n=1
        'Lobular carcinoma with DCIS and LCIS': ['ILC', 'DCIS', 'LCIS'],                                        # n=1
        'Lobular carcinoma with LCIS': ['ILC', 'LCIS'],                                                         # n=3
        'Lobular Carcinoma, NOS': ['ILC', 'IDC'],                                                               # n=1
        'METAPLASTIC CARCINOMA': ['ILC', 'LCIS'],                                                               # n=1
        'Metastatic Adenocarcinoma': ['Metaplastic'],                                                           # n=1
        'Metastatic carcinoma': ['Metastatic'],                                                                 # n=2
        'Metastatic Neuroendocrine Tumour': ['Metastatic'],                                                     # n=1
        'Mixed ductal and lobular invasive carcinoma with DCIS and LCIS':['ILC', 'IDC', 'DCIS', 'LCIS'],        # n=1
        'Mixed IDC & Papillary Carcinoma with DCIS': ['IDC', 'Papillary', 'DCIS'],                              # n=1
        'Mixed IDC and Invasive Mucinous carcinoma with DCIS ': ['IDC', 'Mucinous', 'DCIS'],                    # n=1
        'MIXED INVASIVE DUCTAL CARCINOMA & INVASIVE MUCINOUS CARCINOMA\n': ['IDC', 'Mucinous', 'DCIS'],       # n=1
        'Mixed Invasive Ductal/Lobular Carcinoma with DCIS and LCIS': ['IDC', 'DCIS', 'LCIS'],                  # n=1
        'Mixed Invasive Lobular Carcinoma & Invasive Ductal Carcinoma & DCIS': ['ILC', 'IDC', 'DCIS'],          # n=1
        'Mixed Lobular': ['ILC'],                                                                               # n=1
        'Mixed tubular and ductal with DCIS': ['Tubular', 'IDC', 'DCIS'],                                       # n=1
        'Mucinous Adenocarcinoma': ['Mucinous'],                                                                # n=1
        'Neuroendocrine carcinoma with DCIS': ['Metastatic', 'DCIS'],                                           # n=1
        "PAGET'S DISEASE & INTRADUCTAL CARCINOMA": ['DCIS'],                                                    # n=1
        'Participant rang April 2014 to confirm breast cancer diagnosis August 2013. Recorded on CDMS.':['DNK'],# n=1
        'Phyllodes Tumour': ['Phyllodes'],                                                                      # n=2
        'Pleomorphic Invasive lobular with LCIS': ['ILC', 'LCIS'],                                              # n=1
        'Tubular carcinoma': ['Tubular'],                                                                       # n=3
        'Tubular Carcinoma with DCIS': ['Tubular', 'DCIS'],                                                     # n=3
        'Unknown': ['DNK'],                                                                                     # n=4
        '(blank)': ['DNK'],                                                                                     # n=48
    })
    new_columns = {
        'DCIS': np.zeros([len(column)]),
        'IDC': np.zeros([len(column)]),
        'LCIS': np.zeros([len(column)]),
        'Metastatic': np.zeros([len(column)]),
        'Mucinous': np.zeros([len(column)]),
        'Phyllodes': np.zeros([len(column)]),
        'Papillary': np.zeros([len(column)]),
        'Apocrine': np.zeros([len(column)]),
        'Adenoid Cystic': np.zeros([len(column)]),
        'Metaplastic': np.zeros([len(column)]),
        'Medullary': np.zeros([len(column)]),
        'Tubular': np.zeros([len(column)]),
        'ILC': np.zeros([len(column)]),
        'Invasive Cribriform': np.zeros([len(column)]),
        'DNK': np.zeros([len(column)]),
    }
    for i, entry in enumerate(column):
        try:
            subtypes = path_entries[entry]
        except:
            if np.isnan(entry):
                continue
            else:
                print("Something went wrong")
        for s in subtypes:
            new_columns[s][i] += 1
    return new_columns

float_pattern = re.compile(r"-?\d+(?:\.\d+)?")

def parse_numeric_max(value, data_type):
    """
    Extracts all integers in a messy string and returns the max.
    Examples:
        '8/7' → 8
        'IDC 0 & ILC 4' → 4
        '8 (IDC) 5 (DCIS)' → 8
        None or '' → None
    """
    if value is None:
        return None

    # Ensure string
    s = str(value).strip()

    if s == "":
        return None

    # Extract ints/floats, e.g. "3", "-2", "4.5", "-0.75"
    nums = float_pattern.findall(s)
    if not nums:
        return None

    # Convert to integers and take max
    nums = [float(n) for n in nums]
    return data_type(np.max(nums))

def apply_map(value, col):
    v = "" if value is None else str(value).strip()

    # Case-insensitive direct match
    for k, out in string_mapping[col].items():
        if k.lower() == v.lower():
            return out

    # Fallback if "else" is defined
    if "else" in string_mapping[col]:
        return string_mapping[col]["else"]

    # Otherwise keep original
    return value

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

bad_columns_dic = {
    'IDK what to do with these - please help': {
        'c_LN': 'what do all the 0/# mean',
        'c_ER/PR score': 'what do I do about all the slashes and stuff',
        'c_HER2  score/ comments': 'How to process comments',
        'c_Ki67 %': 'different breasts/tumours',
        'c_HER2  score/ comments2': 'How to process not amplified',
        'c_Ki67 %2': 'ranges/%/>< etc',
    },
}

string_mapping = {
    # general framework
    'column_name': {
        'old_entry': 'replacement',         # n=a
        '': 'replacement_for_blank_entry'   # n=b
    },

    # actual mappings
    'post prev biopsy': { #GE
        '0': 'No',   # n=53462
        '1': 'Yes',   # n=3271
        '2': 'Yes',   # n=17
        '': 'DNK'   # n=1153
    },
    'DiagnosisOfCancer <70': {
        'na': 'No',     # n=15
        'no': 'No',     # n=56140
        'yes': 'Yes',   # n=1747
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
        'else': 'Yes',  # n=515 (it's dates, but should this just be Yes?) todo fix this
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
        'no': 'No',     # n=47883
        'yes': 'Yes',   # n=10019
        '': 'Blank'     # n=1
    },
    'DiagnosisOfCancer': {  # ignore
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
        'Yes': 'DNK',   # n=5
        'both': 'Both', # n=4977
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
        'no': 'No',     # n=7384
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
        'endometrial': 'EC',                                            # n=27
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
    '2+FHno50': {  # GE
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
    'All good factors': {    ########## STOPPED HERE
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
        'no': 'No',     # n=10801
        'yes': 'Yes',   # n=41233
        '': 'Blank',    # n=1
    },
    'AlcoholYN': {
        'DNK': 'DNK',   # n=850
        'no': 'No',     # n=15817
        'yes': 'Yes',   # n=41233
        '': 'Blank',    # n=3
    },
    'MenopausalStatus': {
        'Data not known': 'DNK',            # n=3071
        'Datanot known': 'DNK',             # n=4
        'Not applicable': 'n/a',            # n=1
        'perimenopausal': 'Perimenopausal', # n=10720
        'postmenopausal': 'Postmenopausal', # n=37233
        'premenopausal': 'Premenopausal',   # n=6873
        '': 'Blank',                        # n=1
    },
    'postmen': {
        'no': 'No',     # n=20669
        'yes': 'Yes',   # n=37233
        '': 'Blank',    # n=1
    },
    'HRT': {
        'DNK': 'DNK',   # n=540
        'no': 'No',     # n=36508
        'Yes': 'Yes',   # n=20854
        '': 'Blank',    # n=1
    },
    'HRT2': {
        'DNK': 'DNK',   # n=540
        'no': 'No',     # n=36508
        'Yes': 'Yes',   # n=20854
        '': 'Blank',    # n=1
    },
    'Combined HRT': {
        'unknown': 'DNK',   # n=12286
        'No': 'No',         # n=4008
        'yes': 'Yes',       # n=4567
        'else': 'No',       # n=37041 (a bunch of random patient IDs) todo fix this
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
        'Asian or asian british': 'Asian or Asian British', # n=891
        'Black or black british': 'Black or Black British', # n=671
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
        'no': 'No',     # n=27448
        'Yes': 'Yes',   # n=8142
        'yes': 'Yes',   # n=8142
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
        'incident': 'Incident',     # n=547
        'interval': 'Interval',     # n=630
        'prevalent': 'Prevalent',   # n=458
        '': 'Blank',                # n=56268
    },
    'presumed postmen BC': {
        'no': 'No',     # n=314
        'yes': 'Yes',   # n=1760
        '': 'Blank',    # n=55829
    },
    'Invasive or CIS or both': {
        'both': 'Both',                                                                             # n=1143
        'Both?': 'Both',                                                                            # n=1
        'CIS': 'CIS',                                                                               # n=347
        'Definate cancer confirmed by MR 21/01/15. Op 15/01/15 awaiting pathology report.': 'DNK',  # n=1
        'invasive': 'Invasive',                                                                     # n=517
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
        'Invasive Lobular Carcinoma with LCIS': 'ILC with LCIS',        # n=1
        'Invasive metaplastic carcinoma with DCIS': 'IMC',              # n=1
        'Invasive mucinous carcinoma': 'IMC',                           # n=2
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
        '1.0': '1',                           # n=3
        '2': '2',                           # n=11
        '2.0': '2',                           # n=11
        '3': '3',                           # n=16
        '3.0': '3',                           # n=16
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
        'NEGATIVE': 'Negative',                                             # n=238
        'Negative - 0': 'Negative',                                         # n=8
        'positive': 'Positive',                                             # n=1706
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
        'NEGATIVE': 'Negative',                                             # n=446
        'Negative - 0': 'Negative',                                         # n=14
        'POSITIVE': 'Positive',                                             # n=1494
        'Positive ': 'Positive',                                            # n=7
        'Positive                                                            Positive': 'Positive',  # n=1
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
        'Negative ': 'Negative',                                                                # n=13
        'negative': 'Negative',                                                                 # n=13
        'NEGATIVE': 'Negative',                                                                 # n=13
        'Negative - 1+': 'Negative',                                                            # n=6
        'Negative - 2+ non amplified': 'Negative',                                              # n=15
        'Negative; Negative': 'Negative',                                                       # n=7
        'Not performed': 'Not performed',                                                       # n=38
        'Not reported': 'Not reported',                                                         # n=177
        'Positive': 'Positive',                                                                 # n=1
        'positive': 'Positive',                                                                 # n=1
        'POSITIVE': 'Positive',                                                                 # n=1
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
        'no': 'No',     # n=56998
        'yes': 'Yes',   # n=905
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
    'c_Prev cancer': {
        'No': 'No',             # n=1080
        'Yes': 'Yes',           # n=80
        'Data not known': 'DNK',# n=6
        'Not known': 'DNK',     # n=7
        '': 'DNK',              # n=146
    },
    'c_Cancer diagnosis': {
        'Yes': 'Yes',           # n=2035
        '': 'DNK',              # n=1
    },
    'c_Type of diagnosis - Screen/Inteval (if known)': {
        ' ': 'DNK',                                         # n=116
        'INTERVAL': 'Interval',                             # n=21
        'Self reported by pt on 25/03/2014': 'Interval',    # n=1
        '': 'DNK',                                          # n=1898
    },
    'c_Second diagnosis since joining PROCAS?': {
        'YES - 15/07/2013': 'Yes',  # n=1
        'Yes - 20/10/2014': 'Yes',  # n=1
        'YES - 21/03/2014': 'Yes',  # n=1
        '': 'DNK',                  # n=2033
    },
    'c_Side': {
        'Left': 'Left',         # n=965
        'Left ': 'Left',        # n=24
        'NOT STATED': 'DNK',    # n=1
        'Right': 'Right',       # n=993
        'Unknown': 'DNK',       # n=3
        '': 'DNK',              # n=50
    },
    'c_Multifocal?': {
        ' ': 'No',                          # n=1
        '2 tumours': '2 tumours',           # n=3
        'Bilateral': 'Bilateral',           # n=12
        'Foci x 2 LCIS': '2 tumours',       # n=1
        'Possible': 'Possible',             # n=9
        'Seperate focus of LCIS': 'Yes',    # n=1
        'Yes': 'Yes',                       # n=107
        'Yes (7 foci)': 'Yes',              # n=1
        '': 'No',                           # n=1901
    },
    'c_Invasive or CIS or both': {
        ' ': 'DNK',                                                                                 # n=1
        'Both': 'Both',                                                                             # n=1157
        'Both?': 'Both',                                                                            # n=1
        'CIS': 'CIS',                                                                               # n=346
        'Definate cancer confirmed by MR 21/01/15. Op 15/01/15 awaiting pathology report.': 'DNK',  # n=1
        'Invasive': 'Invasive',                                                                     # n=468
        'Invasive ': 'Invasive',                                                                    # n=1
        '': 'DNK',                                                                                  # n=61
    },
    'c_Grade only invasive': {
        '1': '1',                   # n=230
        '2': '2',                   # n=579
        '2.1': '2',                 # n=1
        '3': '3',                   # n=302
        ' ': 'DNK',                 # n=36
        '1 ': '1',                  # n=103
        '1 (provisional)': '1',     # n=17
        '1 provisional': '1',       # n=1
        '2 ': '2',                  # n=174
        '2 (provisional)': '2',     # n=59
        '3 ': '3',                  # n=91
        '3 (IDC), 2 (IMC)': '3',    # n=1
        '3 (probably)': '3',        # n=1
        '3 (provisional)': '3',     # n=47
        'IDC 3 & ILC 2': '3',       # n=1
        'Not Known': 'DNK',         # n=1
        '': 'DNK',                  # n=392
    },
    'c_DCIS only grade': {
        '3': 'High',                                                        # n=1
        ' ': ' ',                                                           # n=217
        'High': 'High',                                                     # n=419
        'High ': 'High',                                                    # n=4
        'High Grade': 'High',                                               # n=7
        'Intemediate & High': 'High',                                       # n=1
        'Intermedaite': 'Intermediate',                                     # n=1
        'Intermediate': 'Intermediate',                                     # n=353
        'intermediate ': 'Intermediate',                                    # n=1
        'Intermediate & High': 'High',                                      # n=63
        'Intermediate (anterior & posterior) & High (posterior)': 'High',   # n=1
        'Intermediate Grade': 'Intermediate',                               # n=2
        'Intermediate&High Grade': 'High',                                  # n=1
        'Intermediate/High': 'High',                                        # n=142
        'Intermidate & High': 'High',                                       # n=1
        'Low': 'Low',                                                       # n=175
        'Low & High': 'High',                                               # n=1
        'Low & Intermediate': 'Intermediate',                               # n=26
        'Low Grade': 'Low',                                                 # n=1
        'Low, Intermediate & High': 'High',                                 # n=1
        'Low/Intermediate': 'Intermediate',                                 # n=104
        'Low/Intermediate/High': 'High',                                    # n=12
        '': 'DNK',                                                          # n=502

    },
    'c_Vascular invasion': {
        ' ': 'DNK',                                                                 # n=56
        'N': 'No',                                                                  # n=3
        'N/A': 'DNK',                                                               # n=12
        'No': 'No',                                                                 # n=1183
        'No ': 'No',                                                                # n=1
        'No                                                             No': 'No',  # n=1
        'Not Known': 'DNK',                                                         # n=2
        'Not reported': 'DNK',                                                      # n=5
        'Possible': 'Possible',                                                     # n=37
        'Probable': 'Possible',                                                     # n=3
        'Suspicious': 'Possible',                                                   # n=1
        'Uncertain': 'Uncertain',                                                   # n=22
        'Unknown': 'DNK',                                                           # n=1
        'Y': 'Yes',                                                                 # n=3
        'Yes': 'Yes',                                                               # n=247
        'Yes ': 'Yes',                                                              # n=1
        'Yes - Extensive': 'Yes',                                                   # n=1
        '': 'DNK',                                                                  # n=457
    },
    'c_ER  status': {
        ' ': ' ',                                                           # n=5
        'IDC Negative & ILC Positive ': 'Positive',                         # n=1
        'Negative': 'Negative',                                             # n=240
        'Positive': 'Positive',                                             # n=1712
        'Positive ': 'Positive',                                            # n=3
        'Positive                                   Positive': 'Positive',  # n=1
        'Positive, Positive': 'Positive',                                   # n=3
        'Positive`': 'Positive',                                            # n=1
        'Positve': 'Positive',                                              # n=1
        'Postive': 'Positive',                                              # n=3
        'Potitive': 'Positive',                                             # n=1
        'Unable to assess': 'DNK',                                          # n=1
        '': 'DNK',                                                          # n=64
    },
    'c_PR Status': {
        ' ': ' ',                                                                                   # n=5
        'IDC Negative & ILC Positive ': 'Positive',                                                 # n=1
        'Negative': 'Negative',                                                                     # n=448
        'Positive': 'Positive',                                                                     # n=1501
        'Positive ': 'Positive',                                                                    # n=7
        'Positive                                                            Positive': 'Positive', # n=1
        'Postive': 'Positive',                                                                      # n=3
        '': 'DNK',                                                                                  # n=70
    },
    'c_HER2 status': {
        '  ': '  ',                                                                             # n=21
        ' ': ' ',                                                                               # n=1
        'N/A': 'N/A',                                                                           # n=9
        'N/A                                                             Negative': 'Negative', # n=1
        'Nagative': 'Negative',                                                                 # n=1
        'Negative': 'Negative',                                                                 # n=1462
        'Negative ': 'Negative',                                                                # n=13
        'Negative;Negative': 'Negative',                                                        # n=1
        'Not performed': 'Not performed',                                                       # n=7
        'Not reported': 'Not reported',                                                         # n=38
        'Positive': 'Positive',                                                                 # n=179
        'Positive ': 'Positive',                                                                # n=1
        'Positive & Negative': 'Positive',                                                      # n=1
        'Positive(IDC)': 'Positive',                                                            # n=1
        'Positve': 'Positive',                                                                  # n=1
        'Postitive': 'Positive',                                                                # n=1
        'Postive': 'Positive',                                                                  # n=1
        '': 'DNK',                                                                              # n=297
    },
    'c_Side (if different)': {
        'Left': 'Left',     # n=13
        'Right': 'Right',   # n=13
        '': 'Blank',        # n=13
    },
    'c_PATH2': {
        'DCIS': 'DCIS',                                                                                         # n=44
        'DCIS (cribriform)': 'DCIS',                                                                            # n=1
        'DCIS (solid)': 'DCIS',                                                                                 # n=1
        'DCIS with cancerisation of lobules': 'DCIS',                                                           # n=1
        'DCIS with microinvasion': 'DCIS',                                                                      # n=1
        'IDC': 'IDC',                                                                                           # n=54
        'IDC  ': 'IDC',                                                                                         # n=1
        'IDC + DCIS': 'IDC + DCIS',                                                                             # n=2
        'IDC with DCIS': 'IDC + DCIS',                                                                          # n=70
        'IDC with DCIS (cribriform)': 'IDC with DCIS',                                                          # n=1
        'IDC with DCIS (cribriform) Tumour 2': 'IDC + DCIS',                                                    # n=1
        'IDC with DCIS and LCIS': 'IDC + DCIS + LCIS',                                                          # n=2
        'IDC with lobular features and with DCIS': 'IDC + DCIS',                                                # n=1
        'IDC with lobular pattern': 'IDC',                                                                      # n=1
        'IDC with lobular pattern growth with DCIS': 'IDC + DCIS',                                              # n=1
        'ILC': 'ILC',                                                                                           # n=1
        'ILC with LCIS': 'ILC + LCIS',                                                                          # n=1
        'Invasive carcinoma': 'Invasive carcinoma',                                                             # n=3
        'Invasive carcinoma  with DCIS': 'Invasive carcinoma + DCIS',                                           # n=1
        'Invasive ductal with DCIS': 'Invasive ductal with DCIS',                                               # n=1
        'Invasive Lobular Carcinoma': 'Invasive Lobular Carcinoma',                                             # n=8
        'Invasive Lobular Carcinoma ': 'Invasive Lobular Carcinoma',                                            # n=1
        'Invasive lobular carcinoma with DCIS': 'Invasive lobular carcinoma + DCIS',                            # n=1
        'Invasive Lobular Carcinoma with LCIS': 'Invasive Lobular Carcinoma + LCIS',                            # n=30
        'Invasive Lobular Carcinoma with PLCIS': 'Invasive Lobular Carcinoma + PLCIS',                          # n=1
        'Invasive Micropapillary Carcinoma': 'Invasive Micropapillary Carcinoma',                               # n=2
        'Invasive micropapillary carcinoma with DCIS': 'Invasive micropapillary carcinoma + DCIS',              # n=1
        'Invasive Mucinous Carcinoma with DCIS': 'Invasive Mucinous Carcinoma with DCIS',                       # n=1
        'Invasive Mucionous Carcinoma with DCIS': 'Invasive Mucinous Carcinoma with DCIS',                      # n=1
        'Invasive Pleomorphic Lobular Carcinoma': 'Invasive Pleomorphic Lobular Carcinoma',                     # n=1
        'Invasive Pleomorphic Lobular Carcinoma with LCIS': 'Invasive Pleomorphic Lobular Carcinoma + LCIS',    # n=1
        'Invasive tubular +DCIS': 'Invasive tubular + DCIS',                                                    # n=1
        'Invasive Tubular Carcinoma with DCIS': 'Invasive Tubular Carcinoma + DCIS',                            # n=1
        'Invasive with DCIS': 'Invasive + DCIS',                                                                # n=1
        'Invaslive Lobular Carcinoma with LCIS': 'Invasive Lobular Carcinoma + LCIS',                           # n=1
        'Lobular Carcinoma': 'Lobular Carcinoma',                                                               # n=2
        'Mixed ductal and lobular invasive carcinoma with DCIS and LCIS':
            'Mixed ductal and lobular invasive carcinoma + DCIS + LCIS',                                        # n=1
        'Tubular Carcinoma': 'Tubular Carcinoma',                                                               # n=1
        '': 'DNK',                                                                                              # n=1791
    },
    'c_Invasive/ CIS': {
        'Both': 'Both',             # n=121
        'CIS': 'CIS',               # n=48
        'Invasive': 'Invasive',     # n=75
        'Invasive ': 'Invasive',    # n=1
        '': 'DNK',                  # n=1791
    },
    'c_Grade (only invasive)': {
        '1': '1',                   # n=40
        '2': '2',                   # n=110
        '3': '3',                   # n=23
        '30': '3',                  # n=1
        '1 ': '1',                  # n=1
        '1 (provisional)': '1',     # n=1
        '2 (provisional)': '2',     # n=9
        '3 (provisional)': '3',     # n=4
        'Intermediate': '2',        # n=1
        '': 'n/a',                  # n=1846
    },
    'c_DCIS only grade2': {
        'High': 'High',                         # n=38
        'Intermediate': 'Intermediate',         # n=36
        'Intermediate & High': 'High',          # n=5
        'Intermediate / High': 'High',          # n=1
        'Intermediate and high': 'High',        # n=1
        'Intermediate/High': 'High',            # n=11
        'Low': 'Low',                           # n=18
        'Low & Intermediate': 'Intermediate',   # n=4
        'Low/Intermediate': 'Intermediate',     # n=7
        'Low/Intermediate/High': 'High',        # n=1
        '': 'n/a',                              # n=1914
    },
    'c_ER  status2': {
        'Negative': 'Negative',     # n=10
        'Positive': 'Positive',     # n=178
        'Positive ': 'Positive',    # n=3
        'Postive': 'Positive',      # n=1
        '': 'DNK',                  # n=1844
    },
    'c_PR Status2': {
        'Negative': 'Negative',     # n=42
        'Positive': 'Positive',     # n=151
        '': 'n/a',                  # n=1843
    },
    'c_HER2 status2': {
        'Negative ': 'Negative',    # n=1
        'Negaitve ': 'Negative',    # n=158
        'Not Reported': 'DNK',      # n=1
        'Positive': 'Positive',     # n=18
        'Postive': 'Positive',      # n=1
        '': 'n/a',                  # n=1857
    },

    'c_': {

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
    'Age FDR OC': int,                                  # r=(12-94),        b=56771, t=['Data not known']
    'age first prescribed': float,                      # r=(39-75),        b=57587, t=[]
    'BMI 20': float,                                    # r=(8 -94),        b= 9183, t=[]
    'BMI 202': float,                                   # r=(8 -66),        b= 7260, t=[0, inaccurate, #REF!]
    'BMI20grp': int,                                    # r=(1 - 4),        b= 9183, t=[]
    'BMI20grp2': float,                                 # r=(1 - 4),        b= 7718, t=[inaccurate]
    'BMI20Grp3': float,                                 # r=(1 -12),        b= 9185, t=[]
    'Time to DNA': float,                               # r=(0 - 4),        b=48724, t=[]
    'HysterectomyAge': int,                             # r=(18-72),        b=44810, t=['Data not known']
    'OvarianCancerAge': int,                            # r=(23-75),        b=57834, t=[-99]
    'OvarianCancerAge2': int,                           # r=(23-75),        b=57832, t=[-99]
    'AgeAtMenarche': int,                               # r=(5 -75),        b=   15, t=[-99, 0]
    'AgeAtFirstPregnancy': int,                         # r=(14-49),        b=    9, t=[-99, 0]
    'FFTP to age consent': float,                       # r=(1 -39),        b=55717, t=[]
    'agefftp grp': int,                                 # r=(0 - 5),        b=    1, t=[-99]
    'ChildrenNum': int,                                 # r=(0 -15),        b=    1, t=[-99]
    'paritygrp': int,                                   # r=(0 - 4),        b=    1, t=[]
    'OC\'EC from entry': float,                         # r=(-32-7),        b=57738, t=[]
    'mss fh': int,                                      # r=(0 -47),        b=57057, t=[death, FH, new]
    'Density Residual': float,                          # r=(0 - 4),        b= 1147, t=[]
    'InitialTyrerCuzick': float,                        # r=(1 -26),        b=    1, t=[]
    'TC8 only': float,                                  # r=(1 -20),        b=    1, t=[]
    'TC8no wt': float,                                  # r=(1 -23),        b=    1, t=[]
    'v8DR': float,                                      # r=(0 -28),        b=    1, t=[]
    '10 yr avg': float,                                 # r=(0 -30),        b=    1, t=[]
    'TC8 grp': int,                                     # r=(1 - 6),        b=    1, t=[]
    'TCDR': float,                                      # r=(0 -39),        b=    1, t=[#VALUE!, #N/A]
    'TCDRgrp': int,                                     # r=(1 - 5),        b= 1173, t=[]
    'TC8DRgrp2': int,                                   # r=(1 - 6),        b=    2, t=[]
    'DR': float,                                        # r=(0 - 3),        b=    1, t=[]
    'TC8VpDR 10 yr avg': float,                         # r=(0 -30),        b=    1, t=[]
    'DR Volpara': float,                                # r=(0 - 3),        b=    1, t=[]
    'BMI': float,                                       # r=(8 -75),        b=    3, t=[0]
    'BMI grp': int,                                     # r=(1 - 4),        b= 3939, t=[]
    'AgeAtConsent': float,                              # r=(46-84),        b=    1, t=[]
    'age grp': int,                                     # r=(1 - 4),        b=    1, t=[]
    'age grp60': int,                                   # r=(1 - 2),        b=    1, t=[]
    'time from 20': float,                              # r=(26-64),        b=    1, t=[]
    'Age first mammo': float,                           # r=(46-84),        b=    1, t=[]
    'Height_ft': int,                                   # r=(4 - 8),        b=    6, t=[-99, 0]
    'Height_in': int,                                   # r=(0 -12),        b=    6, t=[-99, 13]
    'Heightm': float,                                   # r=(1 - 3),        b=    1, t=[-99, 0]
    'Height group': int,                                # r=(1 - 3),        b= 1166, t=[]
    'Weight_st': int,                                   # r=(5 -28),        b=    9, t=[-99, 0]
    'Weight_lb': int,                                   # r=(5 -28),        b=    9, t=[-99, 0?]
    'WeightKg': float,                                  # r=(35-202),       b=    3, t=[-99, 0]
    'WeightAt20_st': int,                               # r=(4 -59),        b= 7832, t=[Data not known]
    'WeightAt20_lb': int,                               # r=(0 -110),       b=18716, t=[Data not known]
    'WeightAt20_kg': int,                               # r=(25-375),       b= 6285, t=[Data not known]
    'Wtkg20from stlb': float,                           # r=(25-375),       b= 7836, t=[0]
    'ExerciseHoursPerMonth': int,                       # r=(0 -120),       b=15658, t=[Data not known]
    'ExerciseMinsPerMonth': int,                        # r=(0 -130),       b=29649, t=[Data not known]
    'Exercise Grp': int,                                # r=(0 - 4),        b=    1, t=[99]
    'AlcoholUnitsPerWeek': int,                         # r=(0 -1750),      b=18324, t=[Data not known]
    'alc grp': int,                                     # r=(0 - 5),        b=  853, t=[Data not known]
    'alc grp2': int,                                    # r=(0 - 5),        b=  853, t=[]
    'Wt gain': float,                                   # r=(-4- 2),        b= 7865, t=[]
    'Wtgaingrp': int,                                   # r=(1 - 6),        b= 7865, t=[]
    'Wtgaingrp2': int,                                  # r=(1 - 6),        b= 7865, t=[]
    'wt gain per year': float,                          # r=(-0- 0),        b= 7865, t=[#VALUE!]
    'OnHRTYears': int,                                  # r=(0 -55),        b=  189, t=[-99, 0?]
    'OnHRTMonths': int,                                 # r=(0 -60),        b=    7, t=[-99, 0?]
    'age at HRT': float,                                # r=(7 -69),        b=47098, t=[]
    'HRT 10+post 50': float,                            # r=(-3-10),        b=39855, t=[before, few, inter, yes]
    'HRT Last Used (Years)': int,                       # r=(0 -70),        b=  108, t=[-99, 0?]
    'HRT Last Used (Months)': int,                      # r=(0 -18),        b=   39, t=[-99, 0?]
    'StatinsYears': int,                                # r=(0 -65),        b=52539, t=[Data not known]
    'StatinsMonths': int,                               # r=(0 -30),        b=54351, t=[Data not known]
    'statins grp': int,                                 # r=(0 - 3),        b=    1, t=[9, 0?]
    'age at death': float,                              # r=(48-88),        b=55107, t=[]
    'age at censor to 70': float,                       # r=(46-72),        b=    1, t=[]
    'follow up to 70': float,                           # r=(-14-13),       b=    1, t=[]
    'Expected TC8nowt': float,                          # r=(0-0.16),       b=    1, t=[]
    'fu to death': float,                               # r=(0 -12),        b=    1, t=[]
    'Years of follow up': float,                        # r=(0 -13),        b=    1, t=[]
    'Expected TC8DR': float,                            # r=(0-0.33),       b=    1, t=[]
    'expected TC8': float,                              # r=(0-0.21),       b=    1, t=[]
    'Expected TC6': float,                              # r=(0-0.23),       b=    1, t=[]
    'age bc': float,                                    # r=(46-83),        b=55754, t=[]
    'age bc grp': int,                                  # r=(1 - 3),        b=56422, t=[]
    'age BCgrp2': int,                                  # r=(1 - 4),        b=55831, t=[]
    'size': float,                                      # r=(3-120),        b=57903, t=[it's all in mm and some have multiple tumours]
    'grade invasive': int,                              # r=(0 -32),        b=56157, t=[lots of text entry and maybe grade/breast]
    'ER score': int,                                    # r=(0 - 8),        b=55797, t=[lots of text entry and maybe score/breast]
    'PR score': int,                                    # r=(0 - 8),        b=55797, t=[lots of text entry and maybe score/breast]
    'MSS family': int,                                  # r=(0 - 47),       b=56575, t=[]
    'Manchester score proband': int,                    # r=(2 - 6),        b=56377, t=[]
    'path grade': int,                                  # r=(-2 - 2),       b=56697, t=[]
    'Hormonal ER': int,                                 # r=(-3 - 4),       b=56445, t=[]
    'HER2': int,                                        # r=(-6 - 3),       b=56726, t=[]
    'MSS personal': int,                                # r=(-7 -12),       b=56377, t=[]
    'prev BC score': int,                               # r=(2 -13),        b=56997, t=[]
    'total MSS': int,                                   # r=(-5 -48),       b=56379, t=[]
    'VASCombinedAvDensity': float,                      # r=(1 -97),        b= 6418, t=[-99, -43.75]
    'age previous bc': float,                           # r=(24-77),        b=56999, t=[-43.1594798083504]
    'ageprBC grp': int,                                 # r=(1 - 5),        b=56999, t=[]
    'VBD%': float,                                      # r=(1 -35),        b=12939, t=[]
    'FGV cm3': float,                                   # r=(4-349),        b=12939, t=[]
    'age menopauuse': int,                              # r=(20-65),        b=   49, t=[-99, 0]
    'c_Invasive tumour size (mm)': float,               # r=(0-160),        b=  590, t=[]
    'c_Whole tumour  or CIS only SIZE (mm)': float,     # r=(0-160),        b=  296, t=[]
    'c_LN-2': int,                                      # r=(0-3),          b=  426, t=[n/a, Not reported]
    'c_ER score': int,                                  # r=(0-8),          b=   72, t=[num/num, and some text]
    'c_PR score': int,                                  # r=(0-8),          b=   77, t=[num/num, and some text]
    'c_Invasive tumour size (mm)2': int,                # r=(0-110),        b= 1858, t=[]
    'c_Whole tumour  or CIS only SIZE (mm)2': float,    # r=(0-120),        b= 1850, t=[]
    'c_ER score2': int,                                 # r=(0-8),          b= 1844, t=[two different 8s?]
    'c_PR score2': int,                                 # r=(0-8),          b= 1843, t=[num/num]
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
    'c_Study entry date': broken_dates,                 # r=(21/10/2009-18/01/2015), b=    0, t=[mixed format]
    'c_Date of prev cancer': broken_dates,              # r=(  /  /1987-29/11/2010), b=  812, t=[years, date, words]
    'c_Diagnosis date from SCR': broken_dates,          # r=(15/11/2009-06/06/2022), b=    0, t=[mixed format]
}

ignore_column = {
    'c_PATH': extract_subtypes_from_path,               # a bunch of labels with DCIS, LCIS, etc which are broken up
}


if not csv_processed_previously:
    # Read raw CSV from the same folder as script/exe
    # raw_csv_path = resource_path(csv_name)
    # csv_file_pointer = raw_csv_path
    # csv_data = pd.read_csv(csv_file_pointer, sep=',')
    # raw_cancers_path = resource_path(cancers_name)
    # cancers_file_pointer = raw_cancers_path
    # cancers_data = pd.read_csv(cancers_file_pointer, sep=',')
    end_column_of_cancers = 'c_PATH2'

    proCeSsVed = pd.DataFrame(columns=csv_data.columns)
    
    # map csv entries to pre-specified entries
    for col in tqdm(proCeSsVed.columns):
        print("Currently processing column:", col)
        if col in function_mapping:
            proCeSsVed[col] = csv_data[col].apply(function_mapping[col])
        elif col in string_mapping:
            # proCeSsVed[col] = csv_data[col].fillna("").map(string_mapping[col]).fillna(csv_data[col])
            # proCeSsVed[col] = csv_data[col].apply(apply_map)
            proCeSsVed[col] = csv_data[col].apply(
                lambda v: apply_map(v, col)
            )
        elif col in numerical_columns:
            proCeSsVed[col] = csv_data[col].apply(
                lambda v: parse_numeric_max(v, numerical_columns[col])
            )
        else:
            print(f"\n\t Col:{col} does not exist in mapping\n")
            proCeSsVed[col] = csv_data[col]

    proCeSsVed_cancer = pd.DataFrame(columns='c_'+cancers_data.columns)
    # map csv entries to pre-specified entries
    for col in cancers_data.columns:
        print("Currently processing column:", col)
        if 'c_'+col == end_column_of_cancers:
            break
        if 'c_'+col in ignore_column:
            proCeSsVed_cancer['c_'+col] = cancers_data[col]
            continue
        if 'c_'+col in function_mapping:
            proCeSsVed_cancer['c_'+col] = cancers_data[col].apply(function_mapping['c_'+col])
        elif 'c_'+col in string_mapping:
            proCeSsVed_cancer['c_'+col] = cancers_data[col].apply(
                lambda v: apply_map(v, 'c_'+col)
            )
        elif 'c_'+col in numerical_columns:
            proCeSsVed_cancer['c_'+col] = cancers_data[col].apply(
                lambda v: parse_numeric_max(v, numerical_columns['c_'+col])
            )
        else:
            print(f"\n\t Col:{'c_'+col} does not exist in mapping\n")
            proCeSsVed_cancer['c_'+col] = cancers_data[col]

    split_PATH_columns = extract_subtypes_from_path(proCeSsVed_cancer['c_PATH'])
    for column in split_PATH_columns:
        proCeSsVed_cancer[column] = split_PATH_columns[column]

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
        "E+ P+ H+" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "E+ P+ H-" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "E+ P- H+" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "E+ P- H-" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "E- P+ H+" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "E- P+ H-" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "E- P- H+" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "E- P- H-" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "Blank",
        axis=1
    )

    # lobular - ER+ (95%) Her2-(majority)
    # ductal - not lobular?
    proCeSsVed['Lobular or Ductal'] = proCeSsVed.apply(
        lambda r:
        "Lobular (ER+ HER2-)" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "Lobular (ER+ HER2-)" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "Ductal (any other)" if r["HER2 status"] == "Positive" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "Ductal (any other)" if r["HER2 status"] == "Positive" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "Ductal (any other)" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Positive" else
        "Ductal (any other)" if r["HER2 status"] == "Negative" and r["ER status"] == "Positive" and r["PR status"] == "Negative" else
        "Ductal (any other)" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Positive" else
        "Ductal (any other)" if r["HER2 status"] == "Negative" and r["ER status"] == "Negative" and r["PR status"] == "Negative" else
        "Blank",
        axis=1
    )

    # todo
    # **space to add VAS readings from MAI-VAS and MADAM
    # space to add other csvs which contain things like Ki-67
    # click a patient
    # socioeconomic score (from postcode (etc?))

    # Save processed CSV next to script/exe
    processed_path = resource_path(save_name)
    combined_data = pd.merge(
        proCeSsVed,
        proCeSsVed_cancer,
        left_on="ProcID",
        right_on="c_Identifier",
        how="outer"   # inner, left, right, outer
    )
    combined_data.to_csv(processed_path, index=False)

    # Save to new CSV
    combined_data.to_csv(os.path.join(save_dir, save_name), index=False)
else:
    # Just load the preprocessed CSV
    processed_path = resource_path(save_name)
    combined_data = pd.read_csv(processed_path, sep=',')

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
        self.all_cols = cols

        default_x_column = "EthnicOrigin"  # "DOB"
        default_y_column = "Subtype"  # "FGV cm3"  #

        # Choose sensible defaults
        default_x = default_x_column if default_x_column in cols else cols[0]
        if default_y_column in cols and default_y_column != default_x:
            default_y = default_y_column
        else:
            default_y = cols[1] if len(cols) > 1 else cols[0]

        # --- Top control panel ------------------------------------------------
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack(side="top", fill="x")

        # Two horizontal rows inside ctrl
        top_row = ttk.Frame(ctrl)
        top_row.pack(side="top", fill="x")

        bottom_row = ttk.Frame(ctrl)
        bottom_row.pack(side="top", fill="x", pady=(5, 0))

        # --- Top control panel ------------------------------------------------
        # X side
        x_frame = ttk.Frame(top_row)
        x_frame.pack(side="left", padx=(0, 20))

        ttk.Label(x_frame, text="X column:").pack(side="top", anchor="w")

        self.x_search = tk.StringVar()
        x_search_entry = ttk.Entry(x_frame, textvariable=self.x_search, width=25)
        x_search_entry.pack(side="top", fill="x")
        self.x_search.trace_add("write", lambda *args: self.filter_columns("x"))

        self.x_var = tk.StringVar(value=default_x)
        self.x_box = ttk.Combobox(
            x_frame, textvariable=self.x_var, values=self.all_cols,
            state="readonly", width=30
        )
        self.x_box.pack(side="top", fill="x")

        # Y side
        y_frame = ttk.Frame(top_row)
        y_frame.pack(side="left", padx=(0, 20))

        ttk.Label(y_frame, text="Y column:").pack(side="top", anchor="w")

        self.y_search = tk.StringVar()
        y_search_entry = ttk.Entry(y_frame, textvariable=self.y_search, width=25)
        y_search_entry.pack(side="top", fill="x")
        self.y_search.trace_add("write", lambda *args: self.filter_columns("y"))

        self.y_var = tk.StringVar(value=default_y)
        self.y_box = ttk.Combobox(
            y_frame, textvariable=self.y_var, values=self.all_cols,
            state="readonly", width=30
        )
        self.y_box.pack(side="top", fill="x")

        # Swap + Update in the top row
        self.swap_btn = ttk.Button(top_row, text="Swap X/Y", command=self.swap_axes)
        self.swap_btn.pack(side="left", padx=(0, 10))

        self.update_btn = ttk.Button(top_row, text="Update plot", command=self.update_plot)
        self.update_btn.pack(side="left", padx=(0, 20))

        # X outlier controls (top row)
        self.x_use_outlier_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_row,
            text="Filter X outliers",
            variable=self.x_use_outlier_filter,
            command=self.update_plot,
        ).pack(side="left", padx=(10, 2))

        ttk.Label(top_row, text="X Min:").pack(side="left")
        self.x_min_entry = ttk.Entry(top_row, width=6)
        self.x_min_entry.pack(side="left")

        ttk.Label(top_row, text="X Max:").pack(side="left")
        self.x_max_entry = ttk.Entry(top_row, width=6)
        self.x_max_entry.pack(side="left", padx=(0, 10))

        # Y outlier controls
        self.y_use_outlier_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_row,
            text="Filter Y outliers",
            variable=self.y_use_outlier_filter,
            command=self.update_plot,
        ).pack(side="left", padx=(10, 2))

        ttk.Label(top_row, text="Y Min:").pack(side="left")
        self.y_min_entry = ttk.Entry(top_row, width=6)
        self.y_min_entry.pack(side="left")

        ttk.Label(top_row, text="Y Max:").pack(side="left")
        self.y_max_entry = ttk.Entry(top_row, width=6)
        self.y_max_entry.pack(side="left", padx=(0, 10))

        # --- Blank handling ---------------------------------------------------
        self.x_exclude_blank = tk.BooleanVar(value=False)
        self.y_exclude_blank = tk.BooleanVar(value=False)

        # --- CAT×CAT normalisation mode --------------------------------------
        self.cat_norm_mode = tk.StringVar(value="none")  # 'none', 'row', 'col', 'total'

        norm_frame = ttk.Frame(bottom_row)
        norm_frame.pack(side="left", padx=(10, 2))

        ttk.Label(norm_frame, text="CAT×CAT:").pack(side="left")
        ttk.Radiobutton(
            norm_frame, text="Counts",
            value="none", variable=self.cat_norm_mode,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Radiobutton(
            norm_frame, text="Norm X",
            value="row", variable=self.cat_norm_mode,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Radiobutton(
            norm_frame, text="Norm Y",
            value="col", variable=self.cat_norm_mode,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Radiobutton(
            norm_frame, text="Overall",
            value="total", variable=self.cat_norm_mode,
            command=self.update_plot,
        ).pack(side="left")

        # --- Plot mode: 2D vs univariate --------------------------------------
        self.plot_mode = tk.StringVar(value="xy")  # 'xy', 'x', 'y'

        mode_frame = ttk.Frame(bottom_row)
        mode_frame.pack(side="left", padx=(10, 2))

        ttk.Label(mode_frame, text="Plot mode:").pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="X vs Y",
            value="xy", variable=self.plot_mode,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Radiobutton(
            mode_frame, text="X only",
            value="x", variable=self.plot_mode,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Radiobutton(
            mode_frame, text="Y only",
            value="y", variable=self.plot_mode,
            command=self.update_plot,
        ).pack(side="left")

        # Blank filters on bottom row too
        ttk.Checkbutton(
            bottom_row,
            text="Exclude 'Blank' in X",
            variable=self.x_exclude_blank,
            command=self.update_plot,
        ).pack(side="left")

        ttk.Checkbutton(
            bottom_row,
            text="Exclude 'Blank' in Y",
            variable=self.y_exclude_blank,
            command=self.update_plot,
        ).pack(side="left")

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

        # Initial plot
        self.update_plot()

    def filter_columns(self, which):
        """Filter X or Y combobox values based on the search text."""
        if which == "x":
            pattern = self.x_search.get().lower()
            combo = self.x_box
            current = self.x_var.get()
        else:
            pattern = self.y_search.get().lower()
            combo = self.y_box
            current = self.y_var.get()

        if pattern:
            filtered = [c for c in self.all_cols if pattern in c.lower()]
        else:
            filtered = self.all_cols

        combo["values"] = filtered

        # If current selection is not in filtered list, pick first (if any)
        if current not in filtered:
            if filtered:
                combo.set(filtered[0])
            else:
                combo.set("")
        # Optionally refresh the plot when filtering
        # self.update_plot()

    def filter_numeric(self, series, coltype, use_filter_var, min_entry, max_entry):
        """Return a boolean mask for numeric outlier filtering on one axis."""
        # only filter continuous/ordinal; ignore CAT/DATE/LOC
        if coltype not in ["CONT", "ORD"]:
            return pd.Series(True, index=series.index)

        if not use_filter_var.get():
            return pd.Series(True, index=series.index)

        s = pd.to_numeric(series, errors="coerce")
        valid = s.dropna()
        if valid.empty:
            return pd.Series(True, index=series.index)

        min_text = min_entry.get().strip()
        max_text = max_entry.get().strip()

        # default: mean ± 3 * std if no manual bounds
        mean = valid.mean()
        std = valid.std()
        if std is None or np.isnan(std):
            std = 0.0

        if min_text:
            try:
                mn = float(min_text)
            except Exception:
                mn = valid.min()
        else:
            mn = mean - 3 * std

        if max_text:
            try:
                mx = float(max_text)
            except Exception:
                mx = valid.max()
        else:
            mx = mean + 3 * std

        mask = (s >= mn) & (s <= mx)
        return mask.fillna(False)

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
        mode = self.plot_mode.get()

        # ✅ Guard: if requested column(s) invalid, don't try to plot
        if mode == "xy":
            if x_col not in self.df.columns or y_col not in self.df.columns:
                self.fig.clear()
                self.ax = self.fig.add_subplot(111)
                self.ax.text(
                    0.5, 0.5,
                    "Select valid X and Y columns,\nthen press 'Update plot'",
                    ha="center", va="center"
                )
                self.ax.set_axis_off()
                self.canvas.draw()
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert("1.0", "No valid columns selected.")
                return
        else:  # univariate mode
            single_col = x_col if mode == "x" else y_col
            if single_col not in self.df.columns:
                self.fig.clear()
                self.ax = self.fig.add_subplot(111)
                self.ax.text(
                    0.5, 0.5,
                    "Select a valid column,\nthen press 'Update plot'",
                    ha="center", va="center"
                )
                self.ax.set_axis_off()
                self.canvas.draw()
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert("1.0", "No valid column selected.")
                return

        x_full = self.df[x_col]
        y_full = self.df[y_col]

        # --- Univariate mode: X only or Y only -------------------------------
        if mode in ("x", "y"):
            col_name = x_col if mode == "x" else y_col
            s_full = self.df[col_name]
            t = get_type(col_name)

            # build a mask only for this column
            mask = pd.Series(True, index=self.df.index)

            # Blank handling
            if mode == "x" and self.x_exclude_blank.get():
                mask &= (s_full != "Blank")
            if mode == "y" and self.y_exclude_blank.get():
                mask &= (s_full != "Blank")

            # Outlier handling
            if mode == "x":
                mask &= self.filter_numeric(
                    s_full, t,
                    self.x_use_outlier_filter, self.x_min_entry, self.x_max_entry
                )
            else:
                mask &= self.filter_numeric(
                    s_full, t,
                    self.y_use_outlier_filter, self.y_min_entry, self.y_max_entry
                )

            s = s_full[mask]

            # reset figure
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            if hasattr(self, "cbar") and self.cbar is not None:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None

            # numeric-like → histogram
            if t in ["CONT", "ORD", "DATE"]:
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if vals.empty:
                    self.ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
                    self.ax.set_axis_off()
                else:
                    self.ax.hist(vals, bins=30)
                    self.ax.set_xlabel(col_name)
                    self.ax.set_ylabel("Count")

                    if t == "DATE":
                        xticks = self.ax.get_xticks()
                        self.ax.set_xticks(xticks)
                        self.ax.set_xticklabels(
                            [
                                number_to_date(int(round(v))) if not np.isnan(v) else ""
                                for v in xticks
                            ],
                            rotation=45,
                            ha="right",
                        )

            # categorical → barplot
            elif t == "CAT":
                counts = s.value_counts().sort_index()
                if counts.empty:
                    self.ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    self.ax.set_axis_off()
                else:
                    positions = np.arange(len(counts))
                    self.ax.bar(positions, counts.values)
                    self.ax.set_xticks(positions)
                    self.ax.set_xticklabels(
                        [f"{cat}\n(n={n})" for cat, n in counts.items()],
                        rotation=45,
                        ha="right",
                    )
                    self.ax.set_ylabel("Count")
                    self.ax.set_xlabel(col_name)

            else:
                # fallback
                self.ax.text(0.5, 0.5, f"Unsupported type: {t}",
                             ha="center", va="center")
                self.ax.set_axis_off()

            self.fig.tight_layout()
            self.canvas.draw()

            # --- simple univariate stats -------------------------------------
            self.stats_text.delete("1.0", tk.END)
            if t in ["CONT", "ORD", "DATE"]:
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if vals.empty:
                    self.stats_text.insert("1.0", "No numeric data.")
                else:
                    desc = vals.describe()
                    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                        self.stats_text.insert(tk.END, f"{k}: {desc[k]:.4g}\n")
            elif t == "CAT":
                counts = s.value_counts()
                if counts.empty:
                    self.stats_text.insert("1.0", "No data.")
                else:
                    self.stats_text.insert("1.0", f"Unique categories: {len(counts)}\n\n")
                    for cat, n in counts.items():
                        self.stats_text.insert(tk.END, f"{cat}: {n}\n")
            else:
                self.stats_text.insert("1.0", "No univariate stats implemented for this type.")

            return  # 🔚 don't run the 2D logic below

        # build a mask for excluding "Blank"
        mask = pd.Series(True, index=self.df.index)

        # Exclude blanks for CAT-type columns
        if self.x_exclude_blank.get():
            mask &= (x_full != "Blank")
        if self.y_exclude_blank.get():
            mask &= (y_full != "Blank")

        # Apply outlier filtering per axis
        mask &= self.filter_numeric(
            x_full, get_type(x_col),
            self.x_use_outlier_filter, self.x_min_entry, self.x_max_entry
        )
        mask &= self.filter_numeric(
            y_full, get_type(y_col),
            self.y_use_outlier_filter, self.y_min_entry, self.y_max_entry
        )

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

            # remove the single axes and create a 2×2 jointplot-like layout
            self.fig.clear()

            gs = self.fig.add_gridspec(
                2, 2,
                width_ratios=(4, 1),
                height_ratios=(1, 4),
                wspace=0.05,
                hspace=0.05,
            )

            ax_joint = self.fig.add_subplot(gs[1, 0])  # main plot
            ax_histx = self.fig.add_subplot(gs[0, 0], sharex=ax_joint)  # x histogram
            ax_histy = self.fig.add_subplot(gs[1, 1], sharey=ax_joint)  # y histogram

            # --- main joint plot: hexbin ---
            hb = ax_joint.hexbin(
                x, y,
                gridsize=40,
                cmap="viridis",
                mincnt=1,
            )
            self.cbar = self.fig.colorbar(hb, ax=ax_joint, fraction=0.05, pad=0.02)

            ax_joint.set_xlabel(x_col)
            ax_joint.set_ylabel(y_col)

            ax_histx.hist(x.dropna(), bins=30, color="gray")
            ax_histy.hist(y.dropna(), bins=30, orientation="horizontal", color="gray")

            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histx.tick_params(axis="y", left=False)
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histy.tick_params(axis="x", bottom=False)

            # DATE formatting (apply to the correct axis)
            if tx == "DATE":
                xticks = ax_joint.get_xticks()
                ax_joint.set_xticks(xticks)
                ax_joint.set_xticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in xticks
                    ],
                    rotation=45, ha="right",
                )
            if ty == "DATE":
                yticks = ax_joint.get_yticks()
                ax_joint.set_yticks(yticks)
                ax_joint.set_yticklabels(
                    [
                        number_to_date(int(round(t))) if not np.isnan(t) else ""
                        for t in yticks
                    ]
                )

            self.ax = ax_joint

        # 2) CAT (X) vs numeric/DATE (Y): categories on X, numeric on Y
        elif tx == "CAT" and ty in ["CONT", "ORD", "DATE"]:
            df_plot = pd.DataFrame({"cat": x.astype(str), "val": y})

            # collect data per category (ignore categories with no numeric values)
            cats = []
            data = []
            counts = []

            for cat, group in df_plot.groupby("cat"):
                vals = pd.to_numeric(group["val"], errors="coerce").dropna().values
                if len(vals) == 0:
                    continue
                cats.append(cat)
                data.append(vals)
                counts.append(len(vals))

            if len(cats) == 0:
                self.ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
                self.ax.set_axis_off()
            else:
                positions = np.arange(len(cats))

                vp = self.ax.violinplot(
                    data,
                    positions=positions,
                    vert=True,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )

                # axis labels, with n in category labels
                self.ax.set_xlabel(x_col)
                self.ax.set_ylabel(y_col)
                self.ax.set_xticks(positions)
                self.ax.set_xticklabels(
                    [f"{c}\n(n={n})" for c, n in zip(cats, counts)],
                    rotation=45,
                    ha="right",
                )
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

            cats = []
            data = []
            counts = []

            for cat, group in df_plot.groupby("cat"):
                vals = pd.to_numeric(group["val"], errors="coerce").dropna().values
                if len(vals) == 0:
                    continue
                cats.append(cat)
                data.append(vals)
                counts.append(len(vals))

            if len(cats) == 0:
                self.ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
                self.ax.set_axis_off()
            else:
                positions = np.arange(len(cats))

                vp = self.ax.violinplot(
                    data,
                    positions=positions,
                    vert=False,  # horizontal violins
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )

                self.ax.set_xlabel(x_col)   # numeric axis
                self.ax.set_ylabel(y_col)   # categorical axis

                self.ax.set_yticks(positions)
                self.ax.set_yticklabels(
                    [f"{c} (n={n})" for c, n in zip(cats, counts)]
                )
                self.ax.set_title("")
                self.fig.suptitle("")

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
                # row/col totals
                row_totals = table.sum(axis=1)
                col_totals = table.sum(axis=0)
                grand_total = table.values.sum()

                mode = self.cat_norm_mode.get()

                if mode == "row":         # normalise by X (rows)
                    heat = table.div(row_totals.replace(0, np.nan), axis=0)
                    title = "Row-normalised proportion (by X group)"
                elif mode == "col":       # normalise by Y (columns)
                    heat = table.div(col_totals.replace(0, np.nan), axis=1)
                    title = "Column-normalised proportion (by Y group)"
                elif mode == "total":     # overall proportion of total N
                    heat = table / grand_total if grand_total > 0 else table.astype(float)
                    title = "Overall proportion of total"
                else:                     # "none" → raw counts
                    heat = table.astype(float)
                    title = "Count heatmap"

                im = self.ax.imshow(heat.values, aspect="auto")

                # ticks with group sizes
                self.ax.set_xticks(range(len(table.columns)))
                self.ax.set_xticklabels(
                    [f"{col}\n(n={col_totals[col]})" for col in table.columns],
                    rotation=90,
                )

                self.ax.set_yticks(range(len(table.index)))
                self.ax.set_yticklabels(
                    [f"{idx} (n={row_totals[idx]})" for idx in table.index]
                )

                self.ax.set_xlabel(y_col)
                self.ax.set_ylabel(x_col)
                self.ax.set_title(title)

                # overlay counts in each cell (raw n, not normalised)
                max_val = np.nanmax(heat.values)
                if np.isnan(max_val):
                    max_val = 0.0
                thresh = max_val / 2.0 if max_val > 0 else 0

                for i, row_name in enumerate(table.index):
                    for j, col_name in enumerate(table.columns):
                        count = table.iloc[i, j]
                        val = heat.iloc[i, j]
                        colour = "white" if (max_val > 0 and val > thresh) else "black"
                        self.ax.text(
                            j, i,
                            str(count),
                            ha="center",
                            va="center",
                            color=colour,
                            fontsize=30,
                        )

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

        if tx == "CAT" and ty == "CAT":
            stats_results.update(chi2_and_cramers_v(x, y))

        if tx in ["CONT", "ORD", "DATE"] and ty in ["CONT", "ORD", "DATE"]:
            stats_results.update(
                pearson_spearman(
                    pd.to_numeric(x, errors="coerce"),
                    pd.to_numeric(y, errors="coerce"),
                )
            )

        if tx == "CAT" and ty in ["CONT", "ORD", "DATE"]:
            stats_results.update(anova_cat_cont(x, pd.to_numeric(y, errors="coerce")))
        if ty == "CAT" and tx in ["CONT", "ORD", "DATE"]:
            stats_results.update(anova_cat_cont(y, pd.to_numeric(x, errors="coerce")))

        if "LOC" in pair and ("CONT" in pair or "ORD" in pair or "DATE" in pair):
            stats_results.update(
                pearson_spearman(
                    pd.to_numeric(x, errors="coerce"),
                    pd.to_numeric(y, errors="coerce"),
                )
            )

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
    app = DataExplorer(combined_data)
    app.mainloop()

print("Done")
