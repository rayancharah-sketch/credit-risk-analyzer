import numpy as np
import pandas as pd
import streamlit as st

from src.predict import load_model, predict_applicant
from src.train import load_data, clean_data, split_data, DATA_PATH, CATEGORICAL_COLUMNS

LABELS = {
    'person_age': 'Age',
    'person_income': 'Annual income',
    'person_emp_length': 'Years employed',
    'person_home_ownership': 'Home ownership',
    'loan_intent': 'Loan purpose',
    'loan_grade': 'Loan grade',
    'loan_amnt': 'Loan amount',
    'loan_int_rate': 'Interest rate',
    'loan_percent_income': 'Loan-to-income ratio',
    'cb_person_default_on_file': 'Previous default',
    'cb_person_cred_hist_length': 'Credit history length',
}

st.set_page_config(page_title="Credit Risk Analyzer", layout="centered")


# Loaded once and reused on every rerun, instead of on every click
@st.cache_resource
def load_everything():
    model, threshold = load_model()
    x_train, x_test, y_train, y_test = split_data(clean_data(load_data(DATA_PATH)))

    # The average training applicant, in the model's 26-column form — the reference
    # point for explaining what pushes one applicant's risk up or down
    baseline = np.asarray(model.named_steps['preprocessor'].transform(x_train)).mean(axis=0)
    return model, threshold, baseline


# Map a processed column name back to its original feature,
# e.g. 'num__person_income' → 'person_income', 'cat__loan_grade_B' → 'loan_grade'
def original_feature(name):
    name = name.split('__', 1)[1]
    for column in CATEGORICAL_COLUMNS:
        if name.startswith(column + '_'):
            return column
    return name


# Each feature's push on the log-odds of default, relative to the average applicant
def explain(applicant, model, baseline):
    preprocessor = model.named_steps['preprocessor']
    coefficients = model.named_steps['classifier'].coef_[0]

    row = np.asarray(preprocessor.transform(pd.DataFrame([applicant])))[0]
    contributions = coefficients * (row - baseline)

    totals = {}
    for name, value in zip(preprocessor.get_feature_names_out(), contributions):
        feature = original_feature(name)
        totals[feature] = totals.get(feature, 0) + value
    return pd.Series(totals)


def to_nan(value):
    return np.nan if value is None else value


# Human-readable value for the explanation list
def describe(feature, applicant):
    value = applicant[feature]
    if isinstance(value, float) and np.isnan(value):
        return "not provided, typical value assumed"
    if feature in ('person_income', 'loan_amnt'):
        return f"${value:,.0f}"
    if feature == 'loan_int_rate':
        return f"{value:.1f}%"
    if feature == 'loan_percent_income':
        return f"{value:.0%} of income"
    if feature in ('person_emp_length', 'cb_person_cred_hist_length'):
        return f"{value:g} years"
    if feature == 'cb_person_default_on_file':
        return 'yes' if value == 'Y' else 'no'
    return str(value).title() if isinstance(value, str) else f"{value:g}"


model, threshold, baseline = load_everything()

st.title("Credit Risk Analyzer")
st.caption("Loan officer risk check. Loan grade and interest rate come from underwriting.")

with st.form("applicant"):
    st.subheader("Required")
    left, right = st.columns(2)
    income = left.number_input("Annual income ($)", min_value=1.0, value=None, step=1000.0, placeholder="e.g. 55000")
    loan_amount = right.number_input("Loan amount ($)", min_value=1.0, value=None, step=500.0, placeholder="e.g. 10000")
    grade = left.selectbox("Loan grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=None, placeholder="Select a grade")
    interest_rate = right.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=None, step=0.1, placeholder="e.g. 11.5")
    past_default = left.selectbox("Previous default on file", ['N', 'Y'], index=None, placeholder="Select",
                                  format_func=lambda v: 'Yes' if v == 'Y' else 'No')

    st.subheader("Optional")
    st.caption("Leave blank if unknown. The model fills in a typical value from the training data.")
    left, right = st.columns(2)
    age = left.number_input("Age", min_value=18, max_value=100, value=None, step=1)
    emp_length = right.number_input("Years employed", min_value=0.0, max_value=70.0, value=None, step=1.0)
    cred_hist = left.number_input("Credit history length (years)", min_value=0, max_value=60, value=None, step=1)
    home = right.selectbox("Home ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'], index=None, placeholder="Unknown")
    intent = left.selectbox("Loan purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
                                             'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'],
                            index=None, placeholder="Unknown")

    submitted = st.form_submit_button("Check risk", type="primary")

if submitted:
    required = {
        'Annual income': income,
        'Loan amount': loan_amount,
        'Loan grade': grade,
        'Interest rate': interest_rate,
        'Previous default on file': past_default,
    }
    missing = [label for label, value in required.items() if value is None]
    if missing:
        st.error(f"Please fill in: {', '.join(missing)}")
        st.stop()

    applicant = {
        'person_income': income,
        'loan_amnt': loan_amount,
        'loan_percent_income': round(loan_amount / income, 2),
        'loan_grade': grade,
        'loan_int_rate': interest_rate,
        'cb_person_default_on_file': past_default,
        'person_age': to_nan(age),
        'person_emp_length': to_nan(emp_length),
        'cb_person_cred_hist_length': to_nan(cred_hist),
        'person_home_ownership': to_nan(home),
        'loan_intent': to_nan(intent),
    }

    result = predict_applicant(applicant, model, threshold)
    probability = result['default_probability']

    st.divider()
    left, right = st.columns(2)
    left.metric("Probability of default", f"{probability:.1%}")
    right.metric("Recommendation", result['decision'].title())
    st.progress(min(probability, 1.0))

    if result['decision'] == 'REJECT':
        st.error(f"Risk is at or above the {threshold:.0%} threshold, so the recommendation is to reject.")
    else:
        st.success(f"Risk is below the {threshold:.0%} threshold, so the recommendation is to approve.")

    contributions = explain(applicant, model, baseline)
    raises = contributions[contributions > 0].sort_values(ascending=False).head(3)
    lowers = contributions[contributions < 0].sort_values().head(3)

    st.subheader("What drove this score")
    st.caption("Compared with the average applicant in the training data.")
    left, right = st.columns(2)
    left.markdown("**Raises risk**")
    for feature in raises.index:
        left.markdown(f"- {LABELS[feature]}: {describe(feature, applicant)}")
    right.markdown("**Lowers risk**")
    for feature in lowers.index:
        right.markdown(f"- {LABELS[feature]}: {describe(feature, applicant)}")
