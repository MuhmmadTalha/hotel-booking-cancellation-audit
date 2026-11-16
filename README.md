# Hotel Booking Cancellation Audit

A machine-learning project that predicts hotel booking cancellations while examining how data leakage and train/test split choices can inflate model performance.

## Key findings

- Including `reservation_status` and `reservation_status_date` produces perfect scores across all models. These fields leak the outcome and must not be used for prediction.
- In the conservative clean-feature setting, XGBoost achieved:
  - PR-AUC: 0.640
  - ROC-AUC: 0.793
  - Accuracy: 0.743
- A random train/test split increased clean XGBoost PR-AUC from 0.640 to 0.717, showing that random splitting can overestimate real future performance.
- Features such as `country`, `deposit_type`, `assigned_room_type`, and `booking_changes` also increased performance, but their availability at prediction time should be carefully validated.

## Project goal

The goal is not only to predict cancellations, but to answer:

1. Can booking information predict cancellation?
2. Which model performs best using appropriate metrics?
3. How much do data leakage and split strategy inflate apparent performance?

## Dataset

The project uses the Hotel Booking Demand dataset from Kaggle:

`jessemostipak/hotel-booking-demand`

The dataset is downloaded locally with KaggleHub and is intentionally not included in this repository.

## Methodology

- Cleaned missing values, invalid bookings, and duplicates.
- Created date, total-night, and total-guest features.
- Compared three feature sets:
  - `leaky`: includes post-outcome fields, used only to demonstrate leakage.
  - `no_status`: removes definite leakage fields.
  - `clean`: uses a conservative set of features suitable for a realistic prediction setting.
- Compared chronological and random train/test splits.
- Evaluated Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.
- Used PR-AUC, ROC-AUC, precision, recall, F1, and calibration rather than accuracy alone.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
```

Then open `Hotel_Cancellation_Audit.ipynb` in VS Code, select the virtual-environment kernel, and run all cells.

## Outputs

- `audit_results.csv`: results for every split and feature-set comparison.
- `all_results.csv`: combined benchmark results.
- `Hotel_Cancellation_Audit.ipynb`: full analysis, visualizations, model training, and evaluation.

## Limitations

- Results come from one dataset containing two Portuguese hotels.
- A single chronological cutoff does not fully prove that the observed split difference generalizes.
- Some feature availability assumptions should be validated against a real hotel-booking workflow.

## Tools

Python, pandas, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, and KaggleHub.
