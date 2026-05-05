# Liver Transplantation Outcome Prediction

Binary classification of post-transplant survival using a custom missing-data-aware Cost-Sensitive SVM (CS-SVM) pipeline.
Developed as the final project for CS 577 – Python in Machine Learning (Spring 2021), University of Idaho.

> **Note:** The dataset is private clinical data and is not included in this repository.  
> See [`data/README.md`](data/README.md) for the expected data format.

---

## Problem

Predicting patient survival after liver transplantation from pre-operative clinical features.  
This project addresses two core challenges common in medical datasets:

1. **Missing data** — ~27% overall missing rate across features
2. **Class imbalance** — 481 survived vs. 131 did not survive

| Label(Death) | Meaning |
|---|---|
| 0 | Did not survive |
| 1 | Survived |

---

## Approach

### Key challenge: Missing data + Class imbalance

Rather than applying a single global imputation strategy, this project uses a **3-level complete view tree framework** with Cost-Sensitive SVM (CS-SVM) that selects the most appropriate classifier based on each test sample's missing value pattern:

```
Test sample
  │
  ├─ No missing values
  │     → CS-SVM trained on complete cases (Complete View 1)
  │
  ├─ Missing features are rarely missing in training set (≤3 occurrences)
  │     → Level-1 CS-SVM (trained on feature subset excluding those rare columns)
  │
  ├─ Missing features are frequently missing in training set (≥10 occurrences)
  │     → Level-2 CS-SVM (trained on broader feature subset)
  │
  └─ Fallback
        → KNN imputation (k=3) → full-feature CS-SVM
```

### Baselines

Both baselines use KNN imputation (k=3) followed by MinMax scaling applied to the full feature set:

- **Decision Tree** (max depth = 5)
- **Standard SVM** (linear kernel, C=1)

---

## Pipeline

```
Raw CSV
  → Label encoding (RecipientBG, DonorBG)
  → One-hot encoding of multi-label CauseOfBrainDeath (32 binary columns)
  → 70 / 30 stratified train-test split
  → Missing data analysis (identify rare vs. frequent missing features)
  → MinMax scaling (per feature subset)
  → 3-level CS-SVM prediction  ←  proposed method
  → KNN imputation → Decision Tree / CS-SVM  ←  baselines
  → Classification report (precision, recall, F1, per class)
```

---

## Results

Evaluated using `sklearn.metrics.classification_report` with 4 decimal precision.  
The proposed 3-level CS-SVM outperforms both baselines on the minority class (non-survival),
which is the clinically relevant outcome.

---

## Project Structure

```
.
├── src/
│   └── code.py   # Full pipeline: preprocessing, models, evaluation
├── data/
│   └── README.md                  # Data format description (dataset not included)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running the Code

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd liver-transplant-prediction

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Copy your CSV to: data/Namazi_Dataset.csv
#    (See data/README.md for the expected column format)

# 5. Run
python src/code.py
```

---

## Dependencies

| Library | Purpose |
|---|---|
| NumPy | Array operations |
| Pandas | Data loading and indexing |
| scikit-learn | SVM, Decision Tree, KNN imputer, scalers, metrics |
| Matplotlib | Class distribution histograms |

---

## Acknowledgements

- Dataset provided by the Namazi research group (private, clinical)
- Course: CS 577 – Python in Machine Learning, Spring 2021, Computer Science Department, University of Idaho
