# Data

The dataset used in this project (`Namazi_Dataset_ver1.csv`) is a **private clinical dataset**
and is not included in this repository.

To run the code, place your CSV file at `data/Namazi_Dataset_ver1.csv` and update
`DATA_PATH` in `src/liver_transplantation.py` if needed.

---

## Expected Format

A CSV file where each row is one patient record. The columns expected by the
preprocessing pipeline are described below.

| Column index | Name                  | Type              | Notes |
|--------------|-----------------------|-------------------|-------|
| 0            | (categorical)         | int (1-indexed)   | Shifted to 0-indexed |
| 3            | `CauseOfBrainDeath`   | string            | Multi-label, e.g. `"3+7+12"` or `"UC"`. One-hot encoded into 32 binary columns. |
| 7            | `RecipientBG`         | string (blood type) | Label-encoded; missing values allowed |
| 8            | (categorical)         | int (1-indexed)   | Shifted to 0-indexed |
| 9            | (categorical)         | int (2-indexed)   | Shifted to 0-indexed |
| 22           | `CauesofBrainDeath`   | int               | Missing/`.` values treated as NaN |
| 24           | `DonorBG`             | string (blood type) | Label-encoded; missing values allowed |
| 25           | label                 | int (1-indexed)   | Survival outcome. Shifted to 0-indexed (0 = did not survive, 1 = survived) |

- Total features after preprocessing: **56** (original columns minus raw cause column, plus 32 one-hot cause columns)
- Missing values are present and handled by the 3-level strategy in the pipeline
- Approximate dataset size: ~478 patients (70% train / 30% test, stratified)
