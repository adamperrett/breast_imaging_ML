# train_multilabel_tabular.py
import os
import math
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

seed = 27
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------
# CONFIG YOU MUST SET
# --------------------------
csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
full_data = False
if full_data:
    csv_name = 'processed_PROCAS_full_data_with_cancer_data.csv'
else:
    csv_name = 'processed_PROCAS_full_data_only_cancers.csv'
csv_path = os.path.join(csv_directory, csv_name)

csv_data = pd.read_csv(csv_path, low_memory=False)

cancer_subtypes = [
    'DCIS',  # 1391
    'IDC',  # 1356
    # 'LCIS',  # 160
    # 'Metastatic',  # 5
    # 'Mucinous',  # 35
    # 'Phyllodes',  # 2
    # 'Papillary',  # 20
    # 'Apocrine',  # 7
    # 'Adenoid Cystic',  # 1
    # 'Metaplastic',  # 5
    # 'Medullary',  # 2
    # 'Tubular',  # 36
    'ILC',  # 185
    # 'Invasive Cribriform',  # 11
    # 'DNK',  # 5
    # 'no_cancer',  # the rest
]
if full_data:
    cancer_subtypes += ['no_cancer']

numerical_columns = [
    'Date of Entry FHC',
    'BMI 20',
    'BMI20grp',
    'HysterectomyAge',
    'AgeAtMenarche',
    'AgeAtFirstPregnancy', # outliers
    'agefftp grp',
    'ChildrenNum',
    'paritygrp',
    'Density Residual',
    'InitialTyrerCuzick',
    'TC8 only',
    'TC8no wt',
    'v8DR',
    '10 yr avg',
    'TC8 grp',
    'TCDR',
    'TCDRgrp',
    'TC8DRgrp2',
    'DR',
    'TC8VpDR 10 yr avg',
    'DR Volpara',
    'BMI',   # outliers
    'BMI grp',
    'AgeAtConsent',
    'age grp',
    'age grp60',
    'time from 20',
    'DOB',
    'Age first mammo',
    'Height_ft',
    'Height_in',
    'Heightm',
    'Height group',
    'Weight_st',
    'Weight_lb',
    'WeightKg',
    'WeightAt20_st',
    'WeightAt20_lb',
    'WeightAt20_kg',
    'Wtkg20from stlb',
    'BMI 202',
    'BMI20grp2',
    'ExerciseHoursPerMonth',
    'ExerciseMinsPerMonth',
    'Exercise Grp',
    'AlcoholUnitsPerWeek',
    'alc grp',
    'Wt gain',
    'Wtgaingrp',
    'wt gain per year',
    'OnHRTYears',
    'OnHRTMonths',
    'age at HRT',
    'HRT 10+post 50',
    'HRT Last Used (Years)',
    'HRT Last Used (Months)',
    'StatinsYears',
    'alc grp2',
    'StatinsMonths',
    'statins grp',
    'age at censor to 70',
    'follow up to 70',
    'Expected TC8nowt',
    'fu to death',
    'Years of follow up',
    'Expected TC8DR',
    'expected TC8',
    'Expected TC6',
    'BMI20Grp3',
    'Wtgaingrp2',
    'VASCombinedAvDensity',
    'VBD%',
    'FGV cm3',
    'age menopauuse'
]

categorical_columns = [
    'FHC',
    'PROCAS after',
    'panel test',
    'DC study',
    'Hysterectomy',
    'OvariesRemoved',
    'AnyChildrenYN',
    'FDR breast',
    'FDR50',
    '2FDR',
    'SDR',
    '2+FHno50',
    'NICE criteria',
    'NICE BC FDR criteria met',
    'NICE high risk',
    'One FDR <40',
    'Wtage20available',
    'BMI20<21.6',
    'AnyExercise',
    'AlcoholYN',
    'MenopausalStatus',
    'postmen',
    'HRT',
    'HRT pre50',
    'HRT Current?',
    'HRT recent',
    'EthnicOrigin',
    'excluded non white Eur',
    'StatinsEverYN',
    'eligible interval study',
    'Low density',
    'High density VaS',
    'PreviousCancerDiagnosis',
    'HRT2',
    'Combined HRT',
    'early menopause'
]

numerical_cancer_data = [
    'contralateral date',
    'age bc',
    'age bc grp',
    'age BCgrp2',
    'MSS family',
    'Manchester score proband',
    'path grade',
    'MSS personal',
    'total MSS',
    # 'c_Invasive tumour size (mm)',
    # 'c_Whole tumour  or CIS only SIZE (mm)',
    # 'c_Grade only invasive',
    # 'c_Ki67 %'
]

categorical_cancer_data = [
    'post prev biopsy',
    'DiagnosisOfCancer <70',
    'screen detected',
    'Highmod gene',
    'Bilateral',
    'Location',
    'detection',
    'presumed postmen BC',
    'HRTType',
    'HRTName',
    # 'c_Side'
]

if not full_data:
    numerical = numerical_columns + numerical_cancer_data
    categorical = categorical_columns + categorical_cancer_data
else:
    numerical = numerical_columns
    categorical = categorical_columns

'''
process numerical:
    remove outliers
    handle missing values
process categorical:
    one hot encode
    handle missing values
build logistic regression for each class
report most important features for each
'''

# def process_numerical(name, numbers):

def process_numerical(
    series: pd.Series,
    sigma_threshold: float = 4.0,
    impute_strategy: str = "median",
    verbose: bool = True,
    max_sigma: int = 20,
    use_dynamic_sigma: bool = True,
    repeat_len: int = 3,
    dynamic_start_sigma: int = 2,
):
    """
    Preprocess a numeric pandas Series:
      - Reports missing statistics
      - Computes outlier counts for sigma=1..max_sigma
      - Removes outliers either:
          (A) fixed sigma_threshold, or
          (B) dynamic sigma: choose the first sigma > dynamic_start_sigma where the
              outlier count repeats for `repeat_len` consecutive sigma values.
      - Imputes missing values
      - Adds:
          * <col>_imputed_missing_flag  (1 where original value was missing OR set to NaN due to outlier removal and then imputed)
          * <col>_excluded_outlier_flag (1 where value was excluded as an outlier)

    Returns:
        df_out (DataFrame with cleaned column + flags)
        stats  (dict of summary stats)
    """

    col_name = series.name if series.name is not None else "column"

    # Ensure numeric; non-numeric -> NaN
    s0 = pd.to_numeric(series.copy(), errors="coerce")

    n_total = len(s0)
    orig_missing_mask = s0.isna()
    n_missing = int(orig_missing_mask.sum())
    frac_missing = n_missing / n_total if n_total else 0.0

    # Stats before any removal
    mean_val = s0.mean(skipna=True)
    std_val = s0.std(skipna=True)

    # Outlier counts (based on original numeric series)
    outlier_counts = {}
    if std_val and std_val > 0:
        abs_dev = np.abs(s0 - mean_val)
        for sigma in range(1, max_sigma + 1):
            outlier_counts[sigma] = int((abs_dev > sigma * std_val).sum())
    else:
        for sigma in range(1, max_sigma + 1):
            outlier_counts[sigma] = 0

    # Choose sigma for removal
    chosen_sigma = sigma_threshold
    chosen_method = "fixed"

    if use_dynamic_sigma and (std_val and std_val > 0):
        chosen_method = "dynamic"

        # We look for the first run of `repeat_len` identical counts after sigma > dynamic_start_sigma
        run_count = 1
        last_count = None

        # Start checking at (dynamic_start_sigma + 1), but still allow counts to be computed from 1..max_sigma
        for sigma in range(dynamic_start_sigma + 1, max_sigma + 1):
            c = outlier_counts[sigma]
            if last_count is None:
                last_count = c
                run_count = 1
            else:
                if c == last_count:
                    run_count += 1
                else:
                    last_count = c
                    run_count = 1

            if run_count >= repeat_len:
                # Choose the first sigma in this repeating run
                chosen_sigma = sigma - (repeat_len - 1)
                break
        else:
            # If never stabilises, fall back to fixed threshold
            chosen_sigma = sigma_threshold
            chosen_method = "dynamic_fallback_to_fixed"

    # Remove outliers using chosen sigma
    s = s0.copy()
    if std_val and std_val > 0:
        outlier_mask = np.abs(s - mean_val) > chosen_sigma * std_val
        excluded_outlier_flag = outlier_mask.fillna(False).astype(int)
        n_removed = int(outlier_mask.sum())
        s[outlier_mask] = np.nan
    else:
        excluded_outlier_flag = pd.Series(0, index=s.index, dtype=int)
        n_removed = 0

    # Imputation
    if impute_strategy == "mean":
        fill_value = s.mean(skipna=True)
    elif impute_strategy == "median":
        fill_value = s.median(skipna=True)
    else:
        raise ValueError("impute_strategy must be 'mean' or 'median'")

    # Flags
    # imputed_missing_flag: 1 where the final value was imputed (i.e., NaN after outlier removal OR originally missing)
    imputed_missing_flag = s.isna().astype(int)

    # Fill NaNs
    s_imputed = s.fillna(fill_value)

    df_out = pd.DataFrame(
        {
            col_name: s_imputed,
            f"{col_name}_imputed_missing_flag": imputed_missing_flag,
            f"{col_name}_excluded_outlier_flag": excluded_outlier_flag,
        },
        index=series.index,
    )

    stats = {
        "column": col_name,
        "total_n": n_total,
        "missing_n_original": n_missing,
        "missing_fraction_original": frac_missing,
        "mean_before_outlier_removal": float(mean_val) if pd.notna(mean_val) else np.nan,
        "std_before_outlier_removal": float(std_val) if pd.notna(std_val) else np.nan,
        "chosen_sigma": chosen_sigma,
        "chosen_method": chosen_method,
        "outliers_removed": n_removed,
        "outlier_counts_by_sigma_1_to_max": outlier_counts,
        "impute_strategy": impute_strategy,
        "imputed_value_used": float(fill_value) if pd.notna(fill_value) else np.nan,
        "imputed_total_after_outlier_removal": int(imputed_missing_flag.sum()),
    }

    if verbose:
        print(f"\nColumn: {col_name}")
        print(f"Total rows: {n_total}")
        print(f"Missing (original): {n_missing} ({frac_missing:.3%})")
        print(f"Mean (before removal): {mean_val:.5f}" if pd.notna(mean_val) else "Mean (before removal): NaN")
        print(f"Std  (before removal): {std_val:.5f}" if pd.notna(std_val) else "Std  (before removal): NaN")
        print(f"Outlier selection: {chosen_method} (sigma={chosen_sigma})")
        print("Outliers by sigma threshold:")
        for sigma in range(1, max_sigma + 1):
            print(f"  > {sigma}σ : {outlier_counts[sigma]}")
        print(f"Outliers removed: {n_removed}")
        print(f"Imputation strategy: {impute_strategy}")
        print(f"Imputed value used: {fill_value:.5f}" if pd.notna(fill_value) else "Imputed value used: NaN")
        print(f"Total imputed after outlier removal: {int(imputed_missing_flag.sum())}")

    return df_out

def process_categorical(series: pd.Series, verbose: bool = True):
    """
    One-hot encode a categorical pandas Series.
    Missing values are encoded as a separate category called 'missing_value'.

    Returns:
        df_encoded (DataFrame with one-hot encoded columns)
    """

    col_name = series.name if series.name is not None else "column"

    # Count missing
    n_total = len(series)
    n_missing = series.isna().sum()
    frac_missing = n_missing / n_total

    # Replace missing with explicit category
    series_filled = series.fillna("missing_value")

    # Convert to string to avoid mixed types
    series_filled = series_filled.astype(str)

    # One-hot encode
    df_encoded = pd.get_dummies(series_filled, prefix=col_name)

    if verbose:
        print(f"\nColumn: {col_name}")
        print(f"Total rows: {n_total}")
        print(f"Missing values: {n_missing}")
        print(f"Missing fraction: {frac_missing:.3%}")
        print(f"Number of categories (including missing): {df_encoded.shape[1]}")

    return df_encoded


def tune_thresholds_per_label(
    model,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    *,
    grid_min: float = 0.01,
    grid_max: float = 0.99,
    grid_steps: int = 99,
    min_pos: int = 5,
    default_threshold: float = 0.5,
    optimise: str = "f1",
    verbose: bool = True,
):
    """
    Bespoke per-label threshold tuner for imbalanced multi-label OvR models.

    - Uses model.predict_proba(X_val) and tunes a separate threshold for each label.
    - Optimises a simple metric (default: F1) on the validation set.
    - If a label has too few positives (<min_pos) or is degenerate (all 0 or all 1),
      falls back to a sensible default / heuristic.

    Returns:
        thresholds: dict[label -> threshold]
        report_df : per-label tuning summary (for logging / inspection)
    """
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_frame()

    labels = list(y_val.columns)
    proba = model.predict_proba(X_val)
    if proba.shape[1] != len(labels):
        raise ValueError(f"predict_proba returned {proba.shape[1]} columns but y_val has {len(labels)} labels")

    grid = np.linspace(grid_min, grid_max, grid_steps)

    thresholds = {}
    rows = []

    for j, label in enumerate(labels):
        yt = y_val.iloc[:, j].to_numpy().astype(int)
        pp = proba[:, j]

        pos = int(yt.sum())
        n = int(len(yt))
        prev = float(pos / n) if n else 0.0

        note = ""
        best_thr = default_threshold
        best_score = np.nan

        # Degenerate label in this split -> can't tune
        if pos == 0 or pos == n:
            note = "degenerate y in val"
            thresholds[label] = best_thr
            rows.append(
                {"label": label, "positives": pos, "prevalence": prev, "best_thr": best_thr, f"best_{optimise}": best_score, "note": note}
            )
            continue

        # Too few positives -> tuning becomes noise; fallback heuristic
        if pos < min_pos:
            # Heuristic: lower threshold for rarer labels, but keep within bounds
            # (This is intentionally simple and stable.)
            best_thr = float(np.clip(0.5 * prev, 0.01, 0.5))
            note = f"pos<{min_pos} fallback"
            thresholds[label] = best_thr
            rows.append(
                {"label": label, "positives": pos, "prevalence": prev, "best_thr": best_thr, f"best_{optimise}": best_score, "note": note}
            )
            continue

        # Grid search threshold
        best_score = -1.0
        for thr in grid:
            yp = (pp >= thr).astype(int)
            if optimise == "f1":
                score = f1_score(yt, yp, zero_division=0)
            else:
                raise ValueError("Only optimise='f1' is implemented in this bespoke tuner.")

            if score > best_score:
                best_score = score
                best_thr = float(thr)

        thresholds[label] = best_thr
        rows.append(
            {"label": label, "positives": pos, "prevalence": prev, "best_thr": best_thr, f"best_{optimise}": float(best_score), "note": note}
        )

    report_df = pd.DataFrame(rows).sort_values(["prevalence", "positives"], ascending=[True, True])

    if verbose:
        print("\n=== Per-label threshold tuning report (on VAL) ===")
        print(report_df.to_string(index=False))

    return thresholds, report_df


def evaluate_multilabel(model, X, y_true, threshold=0.5, name="VAL"):
    print("\n--------------------------------------\nEvaluating on", name)
    proba = model.predict_proba(X)

    # Accept either scalar threshold or dict[label->thr]
    if isinstance(threshold, dict):
        thr_arr = np.array([threshold.get(lbl, 0.5) for lbl in y_true.columns], dtype=float)
        y_pred = (proba >= thr_arr[None, :]).astype(int)
        threshold_label = "per-label"
    else:
        y_pred = (proba >= float(threshold)).astype(int)
        threshold_label = float(threshold)

    rows = []
    for j, label in enumerate(y_true.columns):
        yt = y_true.iloc[:, j].to_numpy().astype(int)
        yp = y_pred[:, j]
        pp = proba[:, j]

        # Skip metrics that require both classes if a label is all-0 or all-1 in this split
        roc = roc_auc_score(yt, pp) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, pp) if len(np.unique(yt)) > 1 else np.nan
        f1 = f1_score(yt, yp, zero_division=0)

        # --- Confusion Matrix ---
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n--- Confusion Matrix: {label} ---")
        print(f"TN={tn:6d}  FP={fp:6d}")
        print(f"FN={fn:6d}  TP={tp:6d}")

        rows.append({
            "label": label,
            "positives": int(yt.sum()),
            "prevalence": float(yt.mean()),
            "roc_auc": roc,
            "pr_auc": ap,
            "f1@thr": f1,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        })

    df = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    print(f"\n=== {name} per-label metrics (threshold={threshold_label}) ===")
    print(df.to_string(index=False))
    print(f"\n{name} macro PR-AUC: {np.nanmean(df['pr_auc']):.4f}")
    print(f"{name} macro ROC-AUC: {np.nanmean(df['roc_auc']):.4f}")
    return df

def top_features_per_label(ovr_model, feature_names, target_names, top_k=20):
    """
    Returns a dict: label -> dataframe of top +/- features by coefficient magnitude
    """
    results = {}
    for label, est in zip(ovr_model.classes_, ovr_model.estimators_):
        coef = est.coef_.ravel()
        dfc = pd.DataFrame({
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef)
        }).sort_values("abs_coef", ascending=False)

        results[target_names[label]] = dfc.head(top_k)
    return results

def add_odds_multiplier(dfc):
    df = dfc.copy()
    df["odds_multiplier_per_+1"] = np.exp(df["coef"])
    return df

def main():
    skipped_columns = []
    new_columns = []
    label_columns = []
    for col in csv_data.columns:
        if col in numerical:
            new_columns.append(process_numerical(csv_data[col]))
        elif col in categorical:
            new_columns.append(process_categorical(csv_data[col]))
        elif col in cancer_subtypes:
            label_columns.append(csv_data[col])
        else:
            skipped_columns.append(col)
    processed_csv = pd.concat(new_columns, axis=1)
    subtype_targets = pd.concat(label_columns, axis=1)

    print("The following columns were skipped either because they lacked relevance or would lead to data leakage:",
          skipped_columns)

    # --- split into training validation test sets ---
    indexes = np.arange(0, len(processed_csv))
    np.random.shuffle(indexes)
    train_size = 0.7
    val_size = 0.15
    train_indexes = indexes[0: int(len(processed_csv)*train_size)]
    val_indexes = indexes[int(len(processed_csv)*train_size):int(len(processed_csv)*(train_size+val_size))]
    test_indexes = indexes[int(len(processed_csv)*(train_size+val_size)):]

    train_features = processed_csv.iloc[train_indexes].copy()
    train_targets = subtype_targets.iloc[train_indexes].copy()
    val_features = processed_csv.iloc[val_indexes].copy()
    val_targets = subtype_targets.iloc[val_indexes].copy()
    test_features = processed_csv.iloc[test_indexes].copy()
    test_targets = subtype_targets.iloc[test_indexes].copy()

    # --- normalise numerical columns (fit on train only) ---
    scaler = StandardScaler()

    # Fit on training numeric columns
    scaler.fit(train_features[numerical])

    # Transform train/val/test and assign back (preserves other columns unchanged)
    train_features.loc[:, numerical] = scaler.transform(train_features[numerical])
    val_features.loc[:, numerical] = scaler.transform(val_features[numerical])
    test_features.loc[:, numerical] = scaler.transform(test_features[numerical])

    print("Fitting data")
    model = OneVsRestClassifier(
        LogisticRegression(penalty="l2", max_iter=1000, class_weight="balanced")#, solver="liblinear")
    )
    model.fit(train_features, train_targets)

    # --- baseline performance at fixed threshold ---
    val_metrics = evaluate_multilabel(model, val_features, val_targets, threshold=0.5, name="VAL")
    test_metrics = evaluate_multilabel(model, test_features, test_targets, threshold=0.5, name="TEST")

    # --- tune thresholds on validation, then evaluate on validation and test with tuned thresholds ---
    tuned_f1, thr_report = tune_thresholds_per_label(
        model,
        val_features,
        val_targets,
        grid_min=0.01,
        grid_max=0.99,
        grid_steps=99,
        min_pos=5,
        default_threshold=0.5,
        optimise="f1",
        verbose=True
    )

    # Report performance with tuned thresholds on VAL and TEST
    train_metrics_tuned = evaluate_multilabel(model, train_features, train_targets, threshold=tuned_f1, name="TRAIN (tuned f1)")
    val_metrics_tuned = evaluate_multilabel(model, val_features, val_targets, threshold=tuned_f1, name="VAL (tuned f1)")
    test_metrics_tuned = evaluate_multilabel(model, test_features, test_targets, threshold=tuned_f1, name="TEST (tuned f1)")

    # --- Feature importance by label (coefficients) ---
    top_feats = top_features_per_label(model, processed_csv.columns, subtype_targets.columns, top_k=20)

    for label in list(top_feats.keys())[:5]:
        print(f"\n--- Top features for label: {label} ---")
        print(top_feats[label].to_string(index=False))

    # print(add_odds_multiplier(top_feats[train_targets.columns[0]]).head(10).to_string(index=False))

    print("Done")


if __name__ == "__main__":
    main()
