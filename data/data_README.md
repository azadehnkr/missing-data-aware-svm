# Data

The dataset used in this project (`Namazi_Dataset.csv`) is a **private clinical dataset**
and is not included in this repository.

To run the code, place your CSV file at `data/Namazi_Dataset.csv` and update
`DATA_PATH` in `src/liver_transplantation.py` if needed.

---

## Dataset Overview

Clinical and laboratory data for liver transplantation patients collected at Namazi Hospital in Shiraz, Iran, between 2001 and 2009.

| Property | Value |
|---|---|
| Original instances | 818 |
| Instances after preprocessing | 612 |
| Original features | 51 |
| Features after preprocessing | 57 (after one-hot encoding) |
| Overall missing rate | ~27% |
| Train / Test split | 70% / 30% (stratified) |

### Class distribution (imbalanced)

| Label(Death) | Meaning | Count |
|---|---|---|
| 0 | Survived (successful transplant) | 481 |
| 1 | Did not survive | 131 |

> The significant class imbalance (481 vs. 131) is a core challenge this project addresses using Cost-Sensitive SVM (CS-SVM).

---

## Expected Format

A CSV file where each row is one patient record. 

- Total features after preprocessing: **57** (51 original features, minus raw cause column, plus 12 one-hot CauseOfBrainDeath columns + other categorical expansions)
- Missing values are present and handled by the 3-level CS-SVM strategy in the pipeline
- Overall dataset missing rate: **~27%**
- Course: CS 577 – Python in Machine Learning, Spring 2021, University of Idaho
