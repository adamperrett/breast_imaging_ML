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
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


# --------------------------
# CONFIG YOU MUST SET
# --------------------------
csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
full_data = True
if full_data:
    csv_name = 'processed_PROCAS_full_data_with_cancer_data.csv'
else:
    csv_name = 'processed_PROCAS_full_data_only_cancers.csv'
CSV_PATH = os.path.join(csv_directory, csv_name)

TARGET_COLS = [
    "DCIS", "IDC", "LCIS", "Metastatic", "Mucinous", "Phyllodes", "Papillary",
    "Apocrine", "Adenoid Cystic", "Metaplastic", "Medullary", "Tubular", "ILC",
    "Invasive Cribriform", "DNK"
]

# columns to always exclude from features (IDs, timestamps, known leakage, etc.)
IGNORE_COLS = [
    "no_cancer",  # ensure it never becomes a feature even though it's not in TARGET_COLS now
    # "patient_id", "sample_id", "diagnosis_text", ...
]

# If you have multiple rows per patient and must avoid leakage across splits:
GROUP_COL = None  # e.g. "patient_id"

# Splits
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of TRAIN used as validation (so overall val ~ 0.16 if test=0.2)

RANDOM_SEED = 42

# Model selection
USE_LOGISTIC_BASELINE = False  # If True -> Logistic model first (interpretable). If False -> Embedding MLP.

# Training defaults (for the MLP)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.20
DEFAULT_HIDDEN = 512
DEFAULT_LAYERS = 2

# Categorical handling
MAX_UNIQUE_CAT = 20     # drop categorical column if > 20 unique values in TRAINVAL
MIN_CAT_FREQ = 10       # bucket categories with freq < MIN_CAT_FREQ into "__RARE__"
BASE_EMB_DIM = 16       # max embedding dim cap (actual per-col dim is derived)

# Feature importance reporting
TOP_K_FEATURES = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------
# Leakage filtering (clinical / post-diagnosis info)
# --------------------------
# Intentionally narrow: remove columns that are very likely clinical diagnostic outputs (post-diagnosis).
LEAKAGE_CLINICAL_PATTERNS = [
    # pathology / histology / biopsy / staging / grade
    r"\bpath\b", r"pathology", r"histolog", r"biops", r"cytolog",
    r"\btnm\b", r"\bt[_\s-]?stage\b", r"\bn[_\s-]?stage\b", r"\bm[_\s-]?stage\b",
    r"\bgrade\b",
    r"\binvasive\b", r"\bcis\b",

    # explicit pathology content seen leaking
    r"vascular\s*invasion", r"\bvi\b",
    r"lobular", r"ductal",
    r"cribriform", r"tubular", r"mucinous", r"papillary", r"apocrine", r"metaplastic",
    r"medullary", r"phyllodes",

    # lymph node / nodal status (often post-diagnosis)
    r"\bln\b", r"lymph\s*node", r"node\s*status", r"\bnodal\b",

    # receptor / biomarkers / molecular subtype
    r"\ber\b", r"\bpr\b", r"her2", r"ki67",
    r"\ber_pr_her2\b", r"\bsubtype\b", r"triple[-\s]?negative",

    # radiology assessment (strongly tied to the diagnostic pathway)
    r"bi[-\s]?rads", r"birads",

    # detection / pathway fields (diagnosis-context proxies)
    r"screen\s*detected", r"\binterval\b", r"\bprevalent\b", r"\bincident\b",
    r"type\s*of\s*diagnosis", r"detection\b", r"\bdetected\b",

    # explicit diagnosis fields / dates
    r"cancer\s*diagnos", r"dateofcancerdiagnos", r"diagnosis\s*date",
    r"basis\s*diagnosis", r"diagnosis\s*histology",

    # REMOVE diagnosis proxies too (so they can't leak into other labels)
    r"diagnosisofcancer",
    r"diagnosis\s*of\s*cancer",

    # outcome-derived age / previous cancer fields
    r"\bage\s*bc\b",
    r"\bage\s*bc\s*grp\b",
    r"\bage\s*bcgrp\b",
    r"\bage.*previous.*bc\b",
    r"\bageprbc\b",
    r"previous\s*cancer\s*diagnos",
    r"dateofpreviousdiagnos",
    r"date\s*of\s*prev", r"prev\s*cancer",  # catches c_Date of prev cancer, c_Prev cancer

    # follow-up / censoring / death status (future knowledge)
    r"\bstatus\b",
    r"\bfollow\s*up\b", r"\bfu\b",
    r"date\s*last\s*follow",           # "Date last follow up", "Date last follow or death"
    r"dateofdeath", r"\bage\s*at\s*death\b", r"premature\s*death\b",
    r"\bcancer\s*death\b", r"\balcohol\s*death\b", r"\bcvd\s*death\b",
    r"\bdod\b",

    # cohort/clinic proxies + risk score outputs
    r"tyrercuzick",
    r"\btc\d+",                        # TC6/TC8/TC...
    r"\btcd?r\b",                      # TCDR / TCDRgrp
    r"\bv\d+dr\b",                     # v8DR etc
    r"density\s*residual",
    r"expected\s*tc",
    r"tc8", r"tc6",
    r"\bdr\b",                         # DR (will also catch "DR Volpara")

    # chemoprevention fields (often downstream/proxy)
    r"chemoprev", r"prescribed\s*chemoprevention", r"age\s*first\s*prescribed",

    # genetics / clinic workup proxies
    r"\bsnp", r"\bmss\b", r"manchester\s*score", r"panel\s*test", r"highmod\s*gene",

    # identifiers / addresses / locations / linkage keys
    r"\bprocid\b", r"\bassure_.*id\b", r"addressline", r"\blocation\b",
    r"\bidentifier\b",
    r"\btime\s*to\s*dna\b",
]

# Exact-name drops (useful when patterns don't catch a variant or you want to force-remove)
LEAKAGE_EXACT_COLS = {
    # follow-up / status
    "Status",

    # explicit age-at-cancer fields seen as top features
    "age bc", "age bc grp", "age BCgrp2", "ageprBC grp", "age previous bc",

    # bilateral/contralateral (often diagnosis-context / outcome-linked)
    "Bilateral", "contralateral date",

    # chemoprevention
    "Chemoprev Drug", "Prescribed Chemoprevention Date", "age first prescribed",

    # genetics/proxy fields
    "panel test", "Highmod gene", "SNPs", "SNPFHBC", "MSS family", "MSS personal", "Manchester score proband",

    # IDs / linkage keys / address / location
    "ProcID", "ProcID2", "VAS.ProcID", "ASSURE_RAW_ID", "ASSURE_PROCESSED_ANON_ID",
    "AddressLine1", "Location", "Time to DNA",

    # follow-up fields observed in your log
    "Date last follow up", "Date last follow or death",
    "DateOfDeath", "DateOfDeath2", "DateOfDeath3", "dod",
    "DateOfCancerDiagnosis", "DateOfCancerDiagnosis2", "DateOfCancerDiagnosis3",
    "PreviousCancerDiagnosis", "DateOfPreviousDiagnosis",

    # risk-score outputs / derived summaries observed in your log
    "InitialTyrerCuzick", "TC8 only", "TC8no wt", "TC8 grp", "TCDR", "TCDRgrp",
    "Expected TC6", "Expected TC8DR", "Expected TC8nowt", "Expected TC8nowt",
    "DR", "DR Volpara", "Density Residual",

    # diagnosis proxies (remove explicitly)
    "DiagnosisOfCancer",
    "DiagnosisOfCancer2",
    "DiagnosisOfCancer <70",
    "DiagnosisOfCancer<70",

    # specific leaky fields you saw in top features / logs
    "BiRads4",
    "Lobular or Ductal",
    "c_Vascular invasion",
    "c_Type of diagnosis - Screen/Inteval (if known)",
    "screen detected",
    "detection",
}


def apply_clinical_leakage_filter(
    df: pd.DataFrame,
    target_cols: List[str],
    ignore_cols: List[str],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Adds leaky clinical/diagnostic columns to ignore list (doesn't physically drop from df).
    Returns (df, ignore_updated, dropped_cols).
    """
    cols = [c for c in df.columns if c not in target_cols]
    rx = re.compile("|".join(LEAKAGE_CLINICAL_PATTERNS), flags=re.IGNORECASE)

    dropped_by_pattern = [c for c in cols if rx.search(str(c))]
    dropped_by_exact = [c for c in cols if str(c).strip() in LEAKAGE_EXACT_COLS]

    # Also: catch “c_” variants that differ only by spacing/punctuation/case
    # (e.g. "c_Vascular invasion", "c_Vascular invasion2", "c_Vascular invasion (mm)" etc.)
    extra_regex = re.compile(
        r"(vascular\s*invasion|lobular|ductal|bi[-\s]?rads|birads|lymph\s*node|\bln\b|"
        r"screen\s*detected|\binterval\b|\bprevalent\b|\bincident\b|type\s*of\s*diagnosis|detection\b)",
        flags=re.IGNORECASE,
    )
    dropped_by_extra = [c for c in cols if extra_regex.search(str(c))]

    dropped = sorted(set(dropped_by_pattern + dropped_by_exact + dropped_by_extra))
    ignore_updated = sorted(set(ignore_cols + dropped))

    if verbose and dropped:
        print(f"\n[ClinicalLeakage] Dropping {len(dropped)} columns that likely leak/cheat.")
        print("[ClinicalLeakage] Examples:", dropped[:60])

    return df, ignore_updated, dropped


# --------------------------
# Utils
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def coerce_targets_to_float(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Coerce common encodings to {0.0, 1.0}.
    Extend mapping if needed.
    """
    out = df.copy()
    mapping = {
        True: 1.0, False: 0.0,
        "True": 1.0, "False": 0.0,
        "true": 1.0, "false": 0.0,
        "YES": 1.0, "NO": 0.0,
        "Yes": 1.0, "No": 0.0,
        "yes": 1.0, "no": 0.0,
        "Y": 1.0, "N": 0.0,
        "y": 1.0, "n": 0.0,
        1: 1.0, 0: 0.0,
        "1": 1.0, "0": 0.0,
    }
    for c in target_cols:
        out[c] = out[c].map(lambda x: mapping.get(x, x))
        out[c] = pd.to_numeric(out[c], errors="coerce")
        # If NaN in targets, treat as 0 (common in multi-label exports) – adjust if wrong
        out[c] = out[c].fillna(0.0).astype(np.float32)
    return out


def infer_feature_types(df: pd.DataFrame, target_cols: List[str], ignore_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Numeric columns = pandas number dtypes.
    Categorical columns = object/category/bool.
    """
    feature_cols = [c for c in df.columns if c not in target_cols and c not in ignore_cols]

    numeric_cols = []
    cat_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        elif pd.api.types.is_bool_dtype(df[c]):
            cat_cols.append(c)
        else:
            cat_cols.append(c)

    return numeric_cols, cat_cols


# --------------------------
# Preprocessor (fit on TRAIN only)
# --------------------------
@dataclass
class TabularPreprocessor:
    numeric_cols: List[str]
    cat_cols: List[str]
    min_cat_freq: int = MIN_CAT_FREQ

    # numeric
    num_imputer: SimpleImputer = None
    num_qt: QuantileTransformer = None

    # categorical vocab maps (for embeddings / logistic one-hot-ish)
    cat_maps: Dict[str, Dict[str, int]] = None
    cat_vocab_sizes: List[int] = None

    # feature name bookkeeping (for interpretability / logistic)
    numeric_feature_names: List[str] = None
    cat_feature_names: List[str] = None

    def fit(self, df_train: pd.DataFrame):
        # ------------------
        # Numeric: drop cols that are entirely missing in TRAIN
        # ------------------
        if self.numeric_cols:
            all_missing = [c for c in self.numeric_cols if df_train[c].notna().sum() == 0]
            if all_missing:
                print(f"Dropping {len(all_missing)} all-missing numeric cols (train). Example: {all_missing[:10]}")
            self.numeric_cols = [c for c in self.numeric_cols if c not in all_missing]

        # Impute median + add missing indicators
        self.num_imputer = SimpleImputer(strategy="median", add_indicator=True)
        Xn = self.num_imputer.fit_transform(df_train[self.numeric_cols]) if self.numeric_cols else np.empty((len(df_train), 0))

        # Build numeric feature names INCLUDING missing indicators from SimpleImputer
        self.numeric_feature_names = []
        if self.numeric_cols:
            self.numeric_feature_names = list(self.numeric_cols)
            # MissingIndicator features: sklearn stores the indices it added
            # In older sklearn, it's `indicator_.features_`; in newer, it's the same.
            if hasattr(self.num_imputer, "indicator_") and self.num_imputer.indicator_ is not None:
                ind_idx = list(getattr(self.num_imputer.indicator_, "features_", []))
                for j in ind_idx:
                    if 0 <= j < len(self.numeric_cols):
                        self.numeric_feature_names.append(f"{self.numeric_cols[j]}__MISSING")

        # Quantile -> Normal (fit on TRAIN only)
        self.num_qt = QuantileTransformer(
            n_quantiles=min(1000, max(10, Xn.shape[0])),
            output_distribution="normal",
            random_state=RANDOM_SEED,
            subsample=int(1e9),
        )
        if Xn.shape[1] > 0:
            self.num_qt.fit(Xn)

        # ------------------
        # Categorical: build per-column vocab with rare bucketing
        # ------------------
        self.cat_maps = {}
        self.cat_vocab_sizes = []
        self.cat_feature_names = []  # used for logistic (expanded features)

        for c in self.cat_cols:
            s = df_train[c].astype("object")
            s = s.where(s.notna(), "__MISSING__").astype(str)

            vc = s.value_counts(dropna=False)
            s2 = s.map(lambda v: v if vc.get(v, 0) >= self.min_cat_freq else "__RARE__")

            # ids: 0 unused/pad, 1 missing, 2 rare, 3.. known
            vocab = {"__MISSING__": 1, "__RARE__": 2}
            next_id = 3
            for v in sorted(set(s2.values.tolist())):
                if v in vocab:
                    continue
                vocab[v] = next_id
                next_id += 1

            self.cat_maps[c] = vocab
            self.cat_vocab_sizes.append(next_id)

            # For interpretability in the logistic baseline:
            # We will represent each categorical column as one-hot over its vocab (excluding 0).
            # Feature names look like: col=value
            inv = {idx: val for val, idx in vocab.items()}
            for idx in range(1, next_id):
                val = inv.get(idx, "__UNK__")
                self.cat_feature_names.append(f"{c}={val}")

        return self

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_cols:
            return np.empty((len(df), 0), dtype=np.float32)
        Xn = self.num_imputer.transform(df[self.numeric_cols])
        if Xn.shape[1] > 0:
            Xn = self.num_qt.transform(Xn)
        return Xn.astype(np.float32)

    def transform_cats_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Integer IDs per categorical column, shape [N, C]."""
        if not self.cat_cols:
            return np.empty((len(df), 0), dtype=np.int64)

        Xc = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            vocab = self.cat_maps[c]
            s = df[c].astype("object")
            s = s.where(s.notna(), "__MISSING__").astype(str)
            rare_id = vocab["__RARE__"]
            Xc[:, j] = s.map(lambda v: vocab.get(v, rare_id)).values.astype(np.int64)
        return Xc

    def transform_cats_onehot_dense(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dense one-hot expansion of categorical columns using the learned vocab.
        Safe here because MAX_UNIQUE_CAT is small (<=20), so vocab sizes stay small.
        """
        if not self.cat_cols:
            return np.empty((len(df), 0), dtype=np.float32)

        Xc_ids = self.transform_cats_ids(df)  # [N, C]
        parts = []
        for j, c in enumerate(self.cat_cols):
            vs = self.cat_vocab_sizes[j]
            # one-hot over ids 1..vs-1
            # (id 0 unused)
            oh = np.zeros((len(df), vs - 1), dtype=np.float32)
            col_ids = Xc_ids[:, j]
            # map id->index (id 1 -> col 0)
            valid = (col_ids >= 1) & (col_ids < vs)
            oh[np.arange(len(df))[valid], col_ids[valid] - 1] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1) if parts else np.empty((len(df), 0), dtype=np.float32)

    def feature_names_logistic(self) -> List[str]:
        return (self.numeric_feature_names or []) + (self.cat_feature_names or [])


# --------------------------
# Dataset (for MLP)
# --------------------------
class TabDataset(Dataset):
    def __init__(self, Xn: np.ndarray, Xc: np.ndarray, Y: np.ndarray):
        self.Xn = torch.tensor(Xn, dtype=torch.float32)
        self.Xc = torch.tensor(Xc, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.Xn[idx], self.Xc[idx], self.Y[idx]


# --------------------------
# Model (categorical embeddings + MLP)
# --------------------------
class TabEmbedMLP(nn.Module):
    def __init__(
        self,
        num_dim: int,
        cat_vocab_sizes: List[int],
        out_dim: int,
        emb_dim_cap: int = BASE_EMB_DIM,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        dropout: float = DEFAULT_DROPOUT
    ):
        super().__init__()

        self.emb_layers = nn.ModuleList()
        self.emb_dims = []

        for vs in cat_vocab_sizes:
            d = min(emb_dim_cap, max(4, int(round((vs ** 0.25) * 8))))
            self.emb_dims.append(d)
            self.emb_layers.append(nn.Embedding(num_embeddings=vs, embedding_dim=d))

        in_dim = num_dim + sum(self.emb_dims)

        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(d, hidden))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            d = hidden
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        if xc.numel() == 0:
            x = xn
        else:
            embs = []
            for j, emb in enumerate(self.emb_layers):
                embs.append(emb(xc[:, j]))
            x = torch.cat([xn] + embs, dim=1)
        return self.net(x)


# --------------------------
# Metrics & reporting
# --------------------------
def sigmoid_stable_numpy(logits: np.ndarray) -> np.ndarray:
    # stable sigmoid via torch
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def per_label_metrics(y_true: np.ndarray, probs: np.ndarray, label_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Per-label AUROC, AP, prevalence, F1@0.5, and best-F1 threshold (on this split).
    """
    out = {}
    for i, name in enumerate(label_names):
        yi = y_true[:, i]
        pi = probs[:, i]

        prevalence = float(np.mean(yi))
        if np.unique(yi).size < 2:
            out[name] = {
                "prevalence": prevalence,
                "auroc": float("nan"),
                "ap": float("nan"),
                "f1@0.5": float("nan"),
                "best_f1": float("nan"),
                "best_thr": float("nan"),
            }
            continue

        auroc = float(roc_auc_score(yi, pi))
        ap = float(average_precision_score(yi, pi))

        pred05 = (pi >= 0.5).astype(np.int32)
        f1_05 = float(f1_score(yi, pred05, zero_division=0))

        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1.0
        best_thr = 0.5
        for t in thresholds:
            pred = (pi >= t).astype(np.int32)
            f1 = f1_score(yi, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)

        out[name] = {
            "prevalence": prevalence,
            "auroc": auroc,
            "ap": ap,
            "f1@0.5": f1_05,
            "best_f1": float(best_f1),
            "best_thr": float(best_thr),
        }
    return out


def overall_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    aucs = []
    aps = []
    for i in range(y_true.shape[1]):
        yi = y_true[:, i]
        pi = probs[:, i]
        if np.unique(yi).size < 2:
            continue
        aucs.append(roc_auc_score(yi, pi))
        aps.append(average_precision_score(yi, pi))

    metrics = {}
    metrics["macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["macro_ap"] = float(np.mean(aps)) if aps else float("nan")

    pred = (probs >= 0.5).astype(np.int32)
    metrics["micro_f1@0.5"] = float(f1_score(y_true.reshape(-1), pred.reshape(-1), average="micro"))
    metrics["macro_f1@0.5"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return metrics


def print_per_label_table(per_label: Dict[str, Dict[str, float]], title: str, label_names: List[str]):
    print(f"\n{title}")
    for k in label_names:
        m = per_label[k]
        print(
            f"{k:20s} prev={m['prevalence']:.4f} "
            f"auroc={m['auroc']:.4f} ap={m['ap']:.4f} "
            f"f1@0.5={m['f1@0.5']:.4f} best_f1={m['best_f1']:.4f} best_thr={m['best_thr']:.2f}"
        )


# --------------------------
# Logistic baseline (interpretable)
# --------------------------
def train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    label_names: List[str],
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 1024,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Multi-label logistic regression trained in PyTorch (one linear layer).
    Reports per-label metrics and per-label top features by absolute weight.
    """
    model = nn.Linear(X_train.shape[1], y_train.shape[1]).to(DEVICE)

    # imbalance handling (pos_weight)
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pw = (neg + 1.0) / (pos + 1.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_score = -1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            val_probs = sigmoid_stable_numpy(val_logits)
            m = overall_metrics(y_val, val_probs)
            score = m["macro_ap"]  # AP is more meaningful under imbalance
            print(f"LogReg Epoch {ep:03d} | train_loss={np.mean(losses):.4f} | val_macro_ap={m['macro_ap']:.4f} | val_macro_auroc={m['macro_auroc']:.4f}")

            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    val_probs = sigmoid_stable_numpy(val_logits)
    val_metrics = overall_metrics(y_val, val_probs)
    val_per_label = per_label_metrics(y_val, val_probs, label_names)

    # Feature importance (weights) per label
    W = model.weight.detach().cpu().numpy()  # [n_labels, n_features]
    print("\nTop features by |weight| per label (Logistic baseline):")
    for li, lbl in enumerate(label_names):
        w = W[li]
        idx = np.argsort(np.abs(w))[::-1][:TOP_K_FEATURES]
        print(f"\n[{lbl}] Top {TOP_K_FEATURES}:")
        for j in idx:
            print(f"  {feature_names[j]:50s}  w={w[j]: .5f}")

    return model, val_metrics, val_per_label


# --------------------------
# MLP train/eval (optional)
# --------------------------
@torch.no_grad()
def eval_mlp(model: nn.Module, loader: DataLoader, label_names: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()
    all_logits = []
    all_y = []
    for xn, xc, yb in loader:
        xn = xn.to(DEVICE)
        xc = xc.to(DEVICE)
        logits = model(xn, xc)
        all_logits.append(logits.cpu().numpy())
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    probs = sigmoid_stable_numpy(logits)
    return overall_metrics(y, probs), per_label_metrics(y, probs, label_names)


def train_one_run_mlp(
    Xn_train, Xc_train, y_train, Xn_val, Xc_val, y_val,
    cat_vocab_sizes: List[int],
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    dropout=DEFAULT_DROPOUT,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    emb_dim_cap: int = BASE_EMB_DIM,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Dict[str, float]]]:

    train_loader = DataLoader(TabDataset(Xn_train, Xc_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TabDataset(Xn_val, Xc_val, y_val), batch_size=512, shuffle=False)

    model = TabEmbedMLP(
        num_dim=Xn_train.shape[1],
        cat_vocab_sizes=cat_vocab_sizes or [],
        out_dim=y_train.shape[1],
        emb_dim_cap=emb_dim_cap,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    ).to(DEVICE)

    # imbalance handling
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pw = (neg + 1.0) / (pos + 1.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xn, xc, yb in train_loader:
            xn = xn.to(DEVICE)
            xc = xc.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model(xn, xc)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        val_metrics, _ = eval_mlp(model, val_loader, TARGET_COLS)
        score = val_metrics["macro_ap"]
        print(f"MLP Epoch {ep:03d} | train_loss={np.mean(losses):.4f} | val_macro_ap={val_metrics['macro_ap']:.4f} | val_macro_auroc={val_metrics['macro_auroc']:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics, final_val_per_label = eval_mlp(model, val_loader, TARGET_COLS)
    return model, final_val_metrics, final_val_per_label


def random_search_hparams_mlp(
    Xn_train, Xc_train, y_train,
    Xn_val, Xc_val, y_val,
    cat_vocab_sizes: List[int],
    trials: int = 10
):
    best = None
    best_score = -1e9

    for t in range(1, trials + 1):
        hidden = random.choice([128, 256, 512, 768, 1024])
        layers = random.choice([1, 2, 3])
        dropout = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])
        lr = 10 ** random.uniform(-4.0, -2.5)
        wd = 10 ** random.uniform(-6.0, -3.0)
        bs = random.choice([64, 128, 256, 512])
        emb_cap = random.choice([8, 16, 32])

        print(f"\nTrial {t}/{trials}: hidden={hidden}, layers={layers}, dropout={dropout}, "
              f"lr={lr:.2e}, wd={wd:.2e}, bs={bs}, emb_cap={emb_cap}")

        model, val_metrics, _ = train_one_run_mlp(
            Xn_train, Xc_train, y_train,
            Xn_val, Xc_val, y_val,
            cat_vocab_sizes=cat_vocab_sizes,
            hidden=hidden, layers=layers, dropout=dropout,
            lr=lr, weight_decay=wd, batch_size=bs,
            epochs=DEFAULT_EPOCHS,
            emb_dim_cap=emb_cap
        )

        score = val_metrics["macro_ap"]
        print(f"Trial {t} val_macro_ap={score:.4f}")

        if score > best_score:
            best_score = score
            best = (hidden, layers, dropout, lr, wd, bs, emb_cap)

    print("\nBEST HPARAMS:", best, "best_val_macro_ap=", best_score)
    return best


# --------------------------
# main
# --------------------------
def main():
    set_seed(RANDOM_SEED)
    df = pd.read_csv(CSV_PATH, low_memory=False)

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Targets not found in CSV columns: {missing_targets}")

    df = coerce_targets_to_float(df, TARGET_COLS)

    ignore = list(IGNORE_COLS)
    if GROUP_COL is not None and GROUP_COL not in ignore:
        ignore.append(GROUP_COL)

    # Apply narrow leakage exclusion: clinical diagnostic outputs only
    df, ignore, dropped_cols = apply_clinical_leakage_filter(df, TARGET_COLS, ignore, verbose=True)
    print(f"[ClinicalLeakage] Total ignored cols (including leakage): {len(ignore)}")

    numeric_cols, cat_cols = infer_feature_types(df, TARGET_COLS, ignore)

    print(f"Detected {len(numeric_cols)} numeric feature cols, {len(cat_cols)} categorical feature cols.")
    print(f"Targets: {len(TARGET_COLS)}")

    if GROUP_COL is not None:
        raise NotImplementedError("Grouped split requested. Implement GroupShuffleSplit once GROUP_COL is confirmed.")

    # Split first (cardinality computed on trainval only, not test)
    df_trainval, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    df_train, df_val = train_test_split(df_trainval, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    # Drop high-cardinality categorical columns based on TRAINVAL (NOT TEST)
    filtered_cat_cols = []
    dropped = []
    for c in cat_cols:
        nunique = df_trainval[c].nunique(dropna=True)
        if nunique > MAX_UNIQUE_CAT:
            dropped.append((c, int(nunique)))
        else:
            filtered_cat_cols.append(c)

    if dropped:
        dropped_sorted = sorted(dropped, key=lambda x: x[1], reverse=True)
        print(f"Dropping {len(dropped_sorted)} categorical cols with nunique > {MAX_UNIQUE_CAT}. Top 20:")
        for c, nu in dropped_sorted[:20]:
            print(f"  - {c}: nunique={nu}")
    cat_cols = filtered_cat_cols
    print(f"After cardinality filter: {len(cat_cols)} categorical feature cols remain.")

    prep = TabularPreprocessor(numeric_cols=numeric_cols, cat_cols=cat_cols, min_cat_freq=MIN_CAT_FREQ).fit(df_train)

    # Numeric (QT->normal) always used
    Xn_train = prep.transform_numeric(df_train)
    Xn_val = prep.transform_numeric(df_val)
    Xn_test = prep.transform_numeric(df_test)

    y_train = df_train[TARGET_COLS].values.astype(np.float32)
    y_val = df_val[TARGET_COLS].values.astype(np.float32)
    y_test = df_test[TARGET_COLS].values.astype(np.float32)

    if USE_LOGISTIC_BASELINE:
        # Dense one-hot for categoricals (safe because MAX_UNIQUE_CAT is small)
        Xc_train_oh = prep.transform_cats_onehot_dense(df_train)
        Xc_val_oh = prep.transform_cats_onehot_dense(df_val)
        Xc_test_oh = prep.transform_cats_onehot_dense(df_test)

        X_train = np.concatenate([Xn_train, Xc_train_oh], axis=1).astype(np.float32)
        X_val = np.concatenate([Xn_val, Xc_val_oh], axis=1).astype(np.float32)
        X_test = np.concatenate([Xn_test, Xc_test_oh], axis=1).astype(np.float32)

        feature_names = prep.feature_names_logistic()

        print("Shapes:",
              "X_train", X_train.shape, "y_train", y_train.shape,
              "X_val", X_val.shape, "y_val", y_val.shape,
              "X_test", X_test.shape, "y_test", y_test.shape)

        # Train logistic regression baseline
        log_model, val_metrics, val_per_label = train_logistic_baseline(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            label_names=TARGET_COLS,
            lr=0.05,
            weight_decay=1e-4,
            epochs=200,
            batch_size=1024,
        )

        # Test evaluation
        log_model.eval()
        with torch.no_grad():
            test_logits = log_model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        test_probs = sigmoid_stable_numpy(test_logits)
        test_metrics = overall_metrics(y_test, test_probs)
        test_per_label = per_label_metrics(y_test, test_probs, TARGET_COLS)

        print("\nFINAL RESULTS (Logistic baseline)")
        print("Val:", val_metrics)
        print("Test:", test_metrics)

        print_per_label_table(val_per_label, "Per-label metrics (VAL)", TARGET_COLS)
        print_per_label_table(test_per_label, "Per-label metrics (TEST)", TARGET_COLS)

    else:
        # Embedding MLP route (keeps categoricals as IDs)
        Xc_train = prep.transform_cats_ids(df_train)
        Xc_val = prep.transform_cats_ids(df_val)
        Xc_test = prep.transform_cats_ids(df_test)

        print("Shapes:",
              "Xn_train", Xn_train.shape, "Xc_train", Xc_train.shape, "y_train", y_train.shape,
              "Xn_val", Xn_val.shape, "Xc_val", Xc_val.shape, "y_val", y_val.shape,
              "Xn_test", Xn_test.shape, "Xc_test", Xc_test.shape, "y_test", y_test.shape)

        # Optional hyperparam search:
        # best = random_search_hparams_mlp(Xn_train, Xc_train, y_train, Xn_val, Xc_val, y_val, prep.cat_vocab_sizes, trials=10)

        model, val_metrics, val_per_label = train_one_run_mlp(
            Xn_train, Xc_train, y_train,
            Xn_val, Xc_val, y_val,
            cat_vocab_sizes=prep.cat_vocab_sizes
        )

        test_loader = DataLoader(TabDataset(Xn_test, Xc_test, y_test), batch_size=512, shuffle=False)
        test_metrics, test_per_label = eval_mlp(model, test_loader, TARGET_COLS)

        print("\nFINAL RESULTS (Embedding MLP)")
        print("Val:", val_metrics)
        print("Test:", test_metrics)

        print_per_label_table(val_per_label, "Per-label metrics (VAL)", TARGET_COLS)
        print_per_label_table(test_per_label, "Per-label metrics (TEST)", TARGET_COLS)

        # Feature importance for the MLP is not as straightforward as logistic.
        # If you need it, we can add permutation importance on the validation set.


if __name__ == "__main__":
    main()
