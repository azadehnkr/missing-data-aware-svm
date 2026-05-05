# -*- coding: utf-8 -*-
"""
Liver Transplantation Outcome Prediction
CS 505 – Python in Machine Learning, Spring 2021
University of Idaho — Dr. Jamil

Task: Binary classification of post-transplant survival outcome.
Dataset: Namazi_Dataset_ver1.csv (private clinical dataset, not included).

Pipeline:
  1. Preprocessing  — label encoding, one-hot encoding of multi-label cause column
  2. Missing data   — 3-level strategy: feature-subset SVMs + KNN imputation fallback
  3. Baselines      — Decision Tree and standard SVM on KNN-imputed full feature set
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH   = "data/Namazi_Dataset_ver1.csv"   # update path if needed
TEST_SIZE   = 0.30
RANDOM_SEED = 101
SVM_C       = 6
SVM_KERNEL  = "linear"
SVM_WEIGHTS = {0: 0.8, 1: 1}
KNN_NEIGHBORS = 3


# ── 1. Load & preprocess ───────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
le = preprocessing.LabelEncoder()
data = df.values.copy()

# Shift 1-indexed categorical columns to 0-indexed
data[:, 0]  = data[:, 0]  - 1   # column 0
data[:, 8]  = data[:, 8]  - 1   # column 8
data[:, 9]  = data[:, 9]  - 2   # column 9
data[:, 22] = data[:, 22] - 1   # column 22
data[:, 25] = data[:, 25] - 1   # label column

# Encode blood type columns — missing values temporarily set to '0'
data[pd.isnull(df["RecipientBG"]), 7] = "0"
data[:, 7] = le.fit_transform(data[:, 7])

data[pd.isnull(df["DonorBG"]), 24] = "0"
data[:, 24] = le.fit_transform(data[:, 24])
data[data[:, 24] == 0, 24] = np.nan   # restore NaN for missing donor BG

# One-hot encode multi-label CauseOfBrainDeath (values like "3+7+12", "UC")
cause = data[:, 3]
NUM_CAUSES = 32   # causes 1–31 + UC mapped to index 31
N = data.shape[0]
cause_onehot = np.zeros((N, NUM_CAUSES))
for i in range(N):
    parts = str(cause[i]).split("+")
    for part in parts:
        idx = 31 if part == "UC" else int(part) - 1
        cause_onehot[i, idx] = 1

data = np.delete(data, 3, axis=1)           # remove raw cause column
data = np.concatenate((data, cause_onehot), axis=1)

# Fix remaining missing/erroneous values in column 22
data[550, 22] = "0"
data[485, 22] = "0"
data[pd.isnull(df["CauesofBrainDeath"]), 22] = "0"
data[df["CauesofBrainDeath"] == ".", 22]      = "0"
data[:, 22] = data[:, 22].astype(np.int32)
data[data[:, 22] == 0, 22] = np.nan

np.savetxt("data/preprocessed.csv", data, delimiter=",")


# ── 2. Train / test split ──────────────────────────────────────────────────────

label = data[:, 24].astype(np.int32)
data  = np.delete(data, 24, axis=1)

Xtr, Xts, Ytr, Yts = train_test_split(
    data, label, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=label
)
Xtr = Xtr.astype(np.float32)
Xts = Xts.astype(np.float32)
Ytr = Ytr.astype(np.int32)
Yts = Yts.astype(np.int32)


# ── 3. Missing-data analysis ───────────────────────────────────────────────────

# Split training set into complete and incomplete subsets
complete_mask   = np.sum(np.isnan(Xtr), axis=1) == 0
Xtr_complete    = Xtr[complete_mask]
Ytr_complete    = Ytr[complete_mask]
Xtr_incomplete  = Xtr[~complete_mask]
Ytr_incomplete  = Ytr[~complete_mask]

plt.hist(Ytr_complete)
plt.title("Class distribution — complete training samples")
plt.show()

N_FEATURES  = Xtr.shape[1]
all_features = set(range(N_FEATURES))

# mf    : features missing in 1–3 incomplete samples (rare missingness)
# mf_10 : features missing in ≥10 incomplete samples (frequent missingness)
mf    = set()
mf_10 = set()

for i in range(len(Xtr_incomplete)):
    n_missing = int(np.sum(np.isnan(Xtr_incomplete[i, :])))
    missing_cols = set(np.where(np.isnan(Xtr_incomplete[i, :]))[0])
    if 1 <= n_missing <= 3:
        mf |= missing_cols
    if n_missing >= 10:
        mf_10 |= missing_cols

# Level-1 subset: drop rarely-missing features (handle those samples without imputation)
keepFeature_level1  = np.array(sorted(all_features - mf),    dtype=np.int32)
# Level-2 subset: drop frequently-missing features
keepFeature_level2  = np.array(sorted(all_features - mf_10), dtype=np.int32)

Xtr_level1 = Xtr_incomplete[:, keepFeature_level1]
Xtr_level1_complete = Xtr_level1[np.sum(np.isnan(Xtr_level1), axis=1) == 0]
Ytr_level1_complete = Ytr_incomplete[np.sum(np.isnan(Xtr_level1), axis=1) == 0]

Xtr_level2 = Xtr_incomplete[:, keepFeature_level2]
Xtr_level2_complete = Xtr_level2[np.sum(np.isnan(Xtr_level2), axis=1) == 0]
Ytr_level2_complete = Ytr_incomplete[np.sum(np.isnan(Xtr_level2), axis=1) == 0]

plt.hist(Ytr_level1_complete); plt.title("Class distribution — level-1 subset"); plt.show()
plt.hist(Ytr_level2_complete); plt.title("Class distribution — level-2 subset"); plt.show()


# ── 4. Proposed method: 3-level SVM ───────────────────────────────────────────

scaler_complete = MinMaxScaler()
Xtr_complete    = scaler_complete.fit_transform(Xtr_complete)

scaler_level1       = MinMaxScaler()
Xtr_level1_complete = scaler_level1.fit_transform(Xtr_level1_complete)

scaler_level2       = MinMaxScaler()
Xtr_level2_complete = scaler_level2.fit_transform(Xtr_level2_complete)

svm_complete = SVC(kernel=SVM_KERNEL, C=SVM_C, class_weight=SVM_WEIGHTS)
svm_complete.fit(Xtr_complete, Ytr_complete)

svm_level1 = SVC(kernel=SVM_KERNEL, C=SVM_C, class_weight=SVM_WEIGHTS)
svm_level1.fit(Xtr_level1_complete, Ytr_level1_complete)

svm_level2 = SVC(kernel=SVM_KERNEL, C=SVM_C, class_weight=SVM_WEIGHTS)
svm_level2.fit(Xtr_level2_complete, Ytr_level2_complete)

# Predict using the appropriate level for each test sample
Yp_proposed = np.zeros_like(Yts)
for i in range(len(Yts)):
    missing_cols = set(np.where(np.isnan(Xts[i]))[0])
    if len(missing_cols) == 0:
        # No missing values — use full-feature SVM
        Xs = scaler_complete.transform(Xts[i, :].reshape(1, -1))
        Yp_proposed[i] = svm_complete.predict(Xs)
    elif mf.issuperset(missing_cols):
        # Missing features are all rare — use level-1 subset SVM
        Xs = scaler_level1.transform(Xts[i, keepFeature_level1].reshape(1, -1))
        Yp_proposed[i] = svm_level1.predict(Xs)
    elif mf_10.issuperset(missing_cols):
        # Missing features are all frequent — use level-2 subset SVM
        Xs = scaler_level2.transform(Xts[i, keepFeature_level2].reshape(1, -1))
        Yp_proposed[i] = svm_level2.predict(Xs)
    else:
        # Fallback: KNN imputation then full-feature SVM
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        imputer.fit(Xtr, Ytr)
        Xts_im = imputer.transform(Xts[i, :].reshape(1, -1))
        Xs     = scaler_complete.transform(Xts_im)
        Yp_proposed[i] = svm_complete.predict(Xs)

print("=== Proposed Method (3-level SVM) ===")
print(classification_report(Yts, Yp_proposed, digits=4))


# ── 5. Baselines: KNN imputation → Decision Tree & SVM ────────────────────────

imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
Xtr_imp = imputer.fit_transform(Xtr, Ytr)
Xts_imp = imputer.transform(Xts)

scaler  = MinMaxScaler()
Xtr_imp = scaler.fit_transform(Xtr_imp)
Xts_imp = scaler.transform(Xts_imp)

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(Xtr_imp, Ytr)
Yp_tree = tree.predict(Xts_imp)

svm_baseline = SVC(kernel="linear", C=1, class_weight={0: 0.2, 1: 1})
svm_baseline.fit(Xtr_imp, Ytr)
Yp_svm = svm_baseline.predict(Xts_imp)

print("=== Baseline: Decision Tree ===")
print(classification_report(Yts, Yp_tree, digits=4))

print("=== Baseline: SVM (KNN-imputed) ===")
print(classification_report(Yts, Yp_svm, digits=4))
