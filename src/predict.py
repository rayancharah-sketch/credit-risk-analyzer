import joblib
import pandas
import numpy as np

# CONSTANTS
MODEL_PATH = 'models/credit_model.joblib'


def load_model(path=MODEL_PATH):
    # Load the bundle saved by train.py: {'model': ..., 'threshold': ...}
    bundle = joblib.load(path)
    return bundle['model'], bundle['threshold']


def predict_applicant(applicant, model, threshold):
    # The model expects a table (DataFrame), not a single dict.
    # Wrapping the dict in a list → a table with one row.
    applicant_df = pandas.DataFrame([applicant])

    # Same as in train.py: [:, 1] = probability of default. [0] = the first (only) row.
    probability = model.predict_proba(applicant_df)[:, 1][0]

    # Apply the threshold chosen in Phase 6
    decision = 'REJECT' if probability >= threshold else 'APPROVE'

    return {
        'default_probability': round(float(probability), 3),
        'decision': decision,
    }

def ask_number(prompt, required=False, min_value=None):
    while True:
        answer = input(prompt).strip()

        if answer == '':
            if required:
                print("  This field is required.")
                continue
            return np.nan

        try:
            number = float(answer)
        except ValueError:
            print("  Please enter a number.")
            continue

        if min_value is not None and number < min_value:
            print(f"  Must be at least {min_value}.")
            continue

        return number

def ask_choice(prompt, options, required=False):
    # options: the allowed values, e.g. ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    while True:
        answer = input(f"{prompt} {options}: ").strip().upper()   # .upper() → 'rent' works too

        if answer == '':
            if required:
                print("  This field is required.")
                continue
            return np.nan

        # Only accept values the model has seen in training
        if answer in options:
            return answer
        print(f"  Please choose one of: {options}")


if __name__ == "__main__":
    model, threshold = load_model()

    print("Loan Officer Risk Check")
    print("Enter application details. Grade and interest rate come from underwriting.")
    print("Press Enter to skip optional fields.\n")

    # Required fields
    income = ask_number("Annual income: ", required=True, min_value=1)
    loan_amount = ask_number("Loan amount: ", required=True, min_value=1)
    grade = ask_choice("Loan grade (from underwriting)", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], required=True)
    interest_rate = ask_number("Interest rate % (from underwriting): ", required=True, min_value=0)
    past_default = ask_choice("Previous default on file?", ['Y', 'N'], required=True)

    applicant = {
        'person_income': income,
        'loan_amnt': loan_amount,
        'loan_percent_income': round(loan_amount / income, 2),
        'loan_grade': grade,
        'loan_int_rate': interest_rate,
        'cb_person_default_on_file': past_default,

        # Optional fields
        'person_age': ask_number("Age (optional): ", min_value=0),
        'person_emp_length': ask_number("Years employed (optional): ", min_value=0),
        'cb_person_cred_hist_length': ask_number("Credit history length, years (optional): ", min_value=0),
        'person_home_ownership': ask_choice("Home ownership (optional)", ['RENT', 'OWN', 'MORTGAGE', 'OTHER']),
        'loan_intent': ask_choice("Loan purpose (optional)",
                                  ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
                                   'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']),
    }

    result = predict_applicant(applicant, model, threshold)
    print(f"\nDefault probability: {result['default_probability']:.1%}")
    print(f"Decision (threshold {threshold}): {result['decision']}")