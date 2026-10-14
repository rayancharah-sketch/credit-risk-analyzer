# Credit Risk Analyzer

A logistic regression model that estimates the probability a loan will default, built as a **decision-support tool for loan officers**. Given an applicant's financial details and the lender's loan grade, it returns a default probability and an approve/reject recommendation.

**Results:** test ROC-AUC of **0.880** (vs. 0.500 for a no-skill baseline). At the chosen threshold, the model catches **80% of defaulters**.

---

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) on Kaggle (CC0 public domain).

- 32,581 loans, 11 features + 1 target (`loan_status`: 0 = repaid, 1 = defaulted)
- ~22% of loans defaulted, so the classes are imbalanced
- Features cover the applicant (age, income, home ownership, employment length, credit history length, past default) and the loan (amount, purpose, grade, interest rate, loan-to-income ratio)
- **The data is simulated** ("columns simulating credit bureau data"). It is realistic in shape, but these are not real borrowers.

---

## Project structure

```
Credit Risk Analyzer/
├── data/                  # credit_risk_dataset.csv (from Kaggle, CC0 license)
├── models/                # credit_model.joblib (created by train.py)
├── src/
│   ├── eda.py             # exploratory data analysis
│   ├── train.py           # clean → split → build pipeline → tune → evaluate → save model
│   └── predict.py         # load the saved model and score a new applicant
├── requirements.txt
└── README.md
```

- **eda.py**: loads the data and checks its shape, column types, default ratio, missing values, impossible values and duplicates.
- **train.py**: cleans the data, splits it into train and test sets, builds the preprocessing and model pipeline, tunes it, evaluates it and saves it.
- **predict.py**: loads the saved model and asks for an applicant's details in the terminal, then returns the probability of default and a decision.

---

## Approach

### 1. Exploratory data analysis

| Finding                                                         | Action                                                                             |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 165 duplicate rows                                              | Dropped. They could appear in both the train and test sets and inflate the scores. |
| Impossible values (age 144, 123 years employed)                 | Dropped rows with age ≥ 85 or employment ≥ 70 years (8 rows)                       |
| Missing `person_emp_length` (~2.7%) and `loan_int_rate` (~9.6%) | Kept and imputed with the median (inside the pipeline, see below)                  |
| Very high incomes                                               | Kept. Extreme but possible, not errors.                                            |

After cleaning, 32,408 rows remain.

### 2. Train/test split

80/20 split, **stratified** on `loan_status` so both sets keep the ~22% default rate. The test set is used only for the final evaluation.

### 3. Preprocessing pipeline

A scikit-learn `Pipeline` + `ColumnTransformer`, fitted on **training data only** to avoid data leakage:

- **Numeric (7 columns):** median imputation → standard scaling
- **Categorical (4 columns):** most-frequent imputation → one-hot encoding (`handle_unknown="ignore"`)

This turns the 11 raw columns into 26 model inputs. The preprocessor and model are saved together as one object, so new applicants are processed exactly as the training data was.

### 4. Model and tuning

Logistic regression, with the regularization strength `C` tuned by 5-fold cross-validation (`GridSearchCV`, scored on ROC-AUC). The default `C = 1` performed best, and scores were flat for C ≥ 0.1. This suggests the model is not overfitting.

---

## Results

| Model                                                     | ROC-AUC   |
| --------------------------------------------------------- | --------- |
| Dummy baseline (always predicts the average default rate) | 0.500     |
| Logistic regression, cross-validated on training data     | 0.866     |
| **Logistic regression, test set**                         | **0.880** |

The cross-validated and test scores are close, so the model generalizes well to unseen data.

**Test set at threshold 0.2:**

|                        | Predicted repay | Predicted default |
| ---------------------- | --------------- | ----------------- |
| **Actually repaid**    | 4,053           | 1,011             |
| **Actually defaulted** | 283             | 1,135             |

Recall (defaulters caught): **0.80**. Precision (rejections that were real defaulters): **0.53**.

---

## Key decisions

### Choosing the threshold (0.2 instead of 0.5)

The model outputs a probability, and the threshold turns it into a decision. Missing a defaulter (a false negative) costs the lender most of the loan, while rejecting a good applicant (a false positive) costs only the lost interest. Assuming a **7:1 cost ratio** (based on ~11–15% interest and ~70% loss given default), I compared thresholds from 0.10 to 0.70 using **cross-validated predictions on the training data**, so the test set was not used to make the choice.

| Threshold | Recall   | Precision | Cost            |
| --------- | -------- | --------- | --------------- |
| 0.15      | 0.83     | 0.45      | 12,473 (lowest) |
| **0.20**  | **0.79** | **0.51**  | **12,675**      |
| 0.25      | 0.75     | 0.57      | 12,920          |
| 0.50      | 0.55     | 0.76      | 18,874          |

I chose **0.20** instead of the cheapest option (0.15). It is within 2% of the minimum cost and meets two guardrails: recall ≥ 0.75 (protecting the lender) and precision ≥ 0.50 (at least half of rejected applicants would really have defaulted). At 0.15, most rejected applicants would have repaid. The test set confirmed the choice: recall 0.80 and precision 0.53, matching the training estimates.

### Why accuracy is the wrong metric here

Approving everyone gives 78% accuracy without any model. Moving the threshold from 0.5 to 0.2 **lowered** accuracy from 87% to 80%, but it **halved missed defaulters** (607 → 283) and cut the total cost of mistakes by about **33%**. This is why the model is judged on ROC-AUC, recall, precision and cost instead.

### Built for loan officers, not applicants

`loan_grade` and `loan_int_rate` are assigned by the lender's own underwriting process. If applicants entered them, they could simply give themselves an A. So the tool is designed for **loan officers**, who take these values from the lender's system, and the model gives a second opinion on top of the lender's assessment.

---

## What drives risk

Top factors from the model's coefficients (odds ratios, holding the other features constant):

- **Loan grade** is the strongest predictor. Grade G loans have about 126× the odds of default of grade A loans, and risk jumps sharply between grades C and D.
- **Loan-to-income ratio** increases risk: one standard deviation higher gives about 4.2× the odds.
- **Renting** increases risk (about 2.2×), and **owning a home outright** lowers it (about 0.2×).
- **Venture and education loans** default less often than medical or debt-consolidation loans.

**Caveat:** loan amount, income and loan-to-income ratio are correlated, so their individual coefficients can look counterintuitive (e.g. loan amount appears to _lower_ risk once the ratio is accounted for). They should be interpreted together. Similarly, past default (`cb_person_default_on_file`) has little effect because `loan_grade` already captures it.

---

## Limitations and future work

- **Reliance on loan grade:** the model leans heavily on the lender's own risk grade. Without `loan_grade` and `loan_int_rate`, cross-validated AUC drops from 0.866 to **0.804**. An applicant-only model would be a more independent risk check.
- **Fairness:** the model uses `person_age`, a protected attribute under the US Equal Credit Opportunity Act. Error rates have not yet been checked across age groups.
- **Simulated data:** results may not transfer to real lending data.
- **Unverified inputs:** the model trusts what is entered. Real lenders verify income and employment.
- **Cost assumptions:** the 7:1 cost ratio is an estimate. Real costs vary by loan.
- **Other models:** tree-based models (e.g. random forest) could be compared.
- **Feature engineering:** log income, missing-value indicators and ratio features could be tested.
- **Frontend:** a Streamlit dashboard with an applicant form, risk gauge and threshold slider.

---

## Setup

Requires **Python 3.12**.

### First-time setup

1. Create the virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it (pick the command for your terminal):
   | Terminal | Command |
   |---|---|
   | Git Bash | `source .venv/Scripts/activate` |
   | PowerShell | `.venv\Scripts\Activate.ps1` |
   | Command Prompt | `.venv\Scripts\activate.bat` |
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   The dataset is included at `data/credit_risk_dataset.csv` (source: [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), CC0 license), so no download is needed.

### Running the project

Run all commands from the project root, with the venv activated:

```bash
python src/eda.py        # explore the data
python src/train.py      # train, evaluate and save the model to models/
python src/predict.py    # score an applicant (prompts in the terminal)
```

`train.py` must run once before `predict.py`, since it creates the saved model. In `predict.py`, required fields must be filled in, and optional fields can be skipped by pressing Enter (the model fills them with typical values from the training data).
