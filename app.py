import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.predict import load_model, predict_applicant
from src.train import (load_data, clean_data, split_data, DATA_PATH,
                       CATEGORICAL_COLUMNS, FN_COST, FP_COST)

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

# Chart colors (validated for color-vision deficiency)
BLUE = '#2a78d6'     # series 1 / "lowers risk"
ORANGE = '#eb6834'   # series 2
RED = '#e34948'      # "raises risk"
MUTED = '#898781'    # reference lines

st.set_page_config(page_title="Credit Risk Analyzer", layout="centered")


# ---------- Data and model (loaded once, reused on every rerun) ----------

@st.cache_resource
def load_everything():
    model, threshold = load_model()
    df = clean_data(load_data(DATA_PATH))
    x_train, x_test, y_train, y_test = split_data(df)

    # The average training applicant in the model's 26-column form,
    # used as the reference point when explaining one applicant's score
    baseline = np.asarray(model.named_steps['preprocessor'].transform(x_train)).mean(axis=0)
    test_prob = model.predict_proba(x_test)[:, 1]
    return model, threshold, baseline, df, y_test.to_numpy(), test_prob


# Recall, precision, mistakes and cost at every threshold, on the test set
@st.cache_data
def threshold_table(y_true, prob):
    rows = []
    for t in np.round(np.arange(0.05, 0.71, 0.01), 2):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            'threshold': t,
            'Recall': tp / (tp + fn),
            'Precision': tp / (tp + fp) if tp + fp else 0.0,
            'Missed defaulters': fn,
            'Good applicants rejected': fp,
            'Cost': FN_COST * fn + FP_COST * fp,
            'tn': tn, 'tp': tp,
        })
    return pd.DataFrame(rows)


# ---------- Explanation helpers ----------

# 'num__person_income' → 'person_income', 'cat__loan_grade_B' → 'loan_grade'
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


# 'DEBTCONSOLIDATION' → 'Debt consolidation'
def pretty(value):
    value = str(value).title()
    return {'Debtconsolidation': 'Debt consolidation', 'Homeimprovement': 'Home improvement'}.get(value, value)


# Human-readable value for the explanation
def describe(feature, applicant):
    value = applicant[feature]
    if isinstance(value, float) and np.isnan(value):
        return "not provided"
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
    return pretty(value) if isinstance(value, str) else f"{value:g}"


# ---------- Charts ----------

def contribution_chart(contributions, applicant):
    data = pd.DataFrame({
        'Factor': [f"{LABELS[f]}: {describe(f, applicant)}" for f in contributions.index],
        'Strength': contributions.values,
    })
    data['Direction'] = np.where(data['Strength'] > 0, 'Raises risk', 'Lowers risk')
    data = data.sort_values('Strength', ascending=False)

    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=14).encode(
        x=alt.X('Strength:Q', title='← lowers risk     raises risk →', axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('Factor:N', sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color('Direction:N', scale=alt.Scale(domain=['Raises risk', 'Lowers risk'], range=[RED, BLUE]),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['Factor', 'Direction', alt.Tooltip('Strength:Q', format='.2f')],
    )
    zero = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=MUTED).encode(x='x:Q')
    return (bars + zero).properties(height=34 * len(data))


def recall_precision_chart(table, threshold):
    long = table.melt(id_vars='threshold', value_vars=['Recall', 'Precision'],
                      var_name='Metric', value_name='Value')
    hover = alt.selection_point(fields=['threshold'], nearest=True, on='pointerover', empty=False)

    base = alt.Chart(long).encode(
        x=alt.X('threshold:Q', title='Threshold', axis=alt.Axis(format='.0%')),
        y=alt.Y('Value:Q', title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Recall', 'Precision'], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient='top', title=None)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=['Metric', alt.Tooltip('threshold:Q', format='.0%', title='Threshold'),
                 alt.Tooltip('Value:Q', format='.0%')],
    ).add_params(hover)
    end_labels = base.mark_text(align='left', dx=6).encode(
        text='Metric:N', color=alt.value(MUTED),
    ).transform_filter(alt.datum.threshold == table['threshold'].max())
    rule = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color=MUTED).encode(x='threshold:Q')
    return (lines + points + end_labels + rule).properties(height=280)


def cost_chart(table, threshold):
    hover = alt.selection_point(fields=['threshold'], nearest=True, on='pointerover', empty=False)
    base = alt.Chart(table).encode(
        x=alt.X('threshold:Q', title='Threshold', axis=alt.Axis(format='.0%')),
        y=alt.Y('Cost:Q', title='Cost of mistakes', scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(strokeWidth=2, color=BLUE)
    points = base.mark_point(size=70, filled=True, color=BLUE).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('threshold:Q', format='.0%', title='Threshold'), alt.Tooltip('Cost:Q', format=',')],
    ).add_params(hover)
    best = table.loc[[table['Cost'].idxmin()]]
    best_point = alt.Chart(best).mark_point(size=90, filled=True, color=BLUE).encode(x='threshold:Q', y='Cost:Q')
    best_label = alt.Chart(best).mark_text(dy=-14, color=MUTED).encode(
        x='threshold:Q', y='Cost:Q', text=alt.value('Lowest cost'))
    rule = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color=MUTED).encode(x='threshold:Q')
    return (line + points + best_point + best_label + rule).properties(height=240)


def roc_chart(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    data = pd.DataFrame({'False positive rate': fpr, 'True positive rate': tpr})
    curve = alt.Chart(data).mark_line(strokeWidth=2, color=BLUE).encode(
        x=alt.X('False positive rate:Q', axis=alt.Axis(format='.0%')),
        y=alt.Y('True positive rate:Q', axis=alt.Axis(format='.0%')),
        tooltip=[alt.Tooltip('False positive rate:Q', format='.1%'), alt.Tooltip('True positive rate:Q', format='.1%')],
    )
    diagonal = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color=MUTED, strokeWidth=1).encode(x='x:Q', y='y:Q')
    return (diagonal + curve).properties(height=320)


def default_rate_chart(df, column, order=None, horizontal=False):
    data = (df.groupby(column, observed=True)['loan_status']
              .agg(['mean', 'size']).reset_index()
              .rename(columns={'mean': 'Default rate', 'size': 'Loans'}))
    data[column] = data[column].map(pretty)
    if order is None:
        order = data.sort_values('Default rate', ascending=False)[column].tolist()

    category = alt.X(f'{column}:N', sort=order, title=None, axis=alt.Axis(labelAngle=0))
    rate = alt.Y('Default rate:Q', axis=alt.Axis(format='.0%'))
    if horizontal:
        category = alt.Y(f'{column}:N', sort=order, title=None, axis=alt.Axis(labelLimit=200))
        rate = alt.X('Default rate:Q', axis=alt.Axis(format='.0%'))
        mark = alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4, size=16)
    else:
        mark = alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4)

    chart = mark.encode(
        category, rate,
        tooltip=[alt.Tooltip(f'{column}:N', title=LABELS.get(column, column)),
                 alt.Tooltip('Default rate:Q', format='.1%'), alt.Tooltip('Loans:Q', format=',')],
    ).properties(height=240)
    return chart, data


# ---------- Page ----------

model, threshold, baseline, df, y_test, test_prob = load_everything()

st.title("Credit Risk Analyzer")
st.caption("Loan officer risk check. Loan grade and interest rate come from underwriting.")

tab_check, tab_model, tab_data = st.tabs(["Risk check", "Model performance", "Data insights"])

# ----- Tab 1: score one applicant -----
with tab_check:
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

    if submitted and not missing:
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
        st.subheader("What drove this score")
        st.caption("Each bar shows how a factor moves this applicant's risk compared with the average "
                   "applicant in the training data. Fields marked \"not provided\" were filled with a typical value.")
        st.altair_chart(contribution_chart(contributions, applicant))

# ----- Tab 2: model performance and threshold -----
with tab_model:
    table = threshold_table(y_test, test_prob)

    left, right = st.columns(2)
    left.metric("Model ROC-AUC", f"{roc_auc_score(y_test, test_prob):.3f}")
    right.metric("No-skill baseline", "0.500")
    st.caption(f"All results on this page use the held-out test set ({len(y_test):,} loans) that the model never saw in training.")

    st.subheader("Try a different threshold")
    st.caption(f"The saved model uses {threshold:.0%}, chosen with cross-validation on the training data. "
               "Move the slider to see the trade-off.")
    chosen = st.slider("Reject when risk is at least", 0.05, 0.70, float(threshold), 0.01, format="%.2f")
    row = table.loc[(table['threshold'] - chosen).abs().idxmin()]

    a, b, c = st.columns(3)
    a.metric("Defaulters caught", f"{row['Recall']:.0%}")
    b.metric("Missed defaulters", f"{int(row['Missed defaulters']):,}")
    c.metric("Good applicants rejected", f"{int(row['Good applicants rejected']):,}")
    a, b, c = st.columns(3)
    a.metric("Precision", f"{row['Precision']:.0%}")
    b.metric("Cost of mistakes", f"{int(row['Cost']):,}")
    c.metric("Cost ratio (miss : reject)", f"{FN_COST} : {FP_COST}")

    st.markdown("**Recall and precision by threshold**")
    st.altair_chart(recall_precision_chart(table, chosen))

    st.markdown("**Cost of mistakes by threshold**")
    st.altair_chart(cost_chart(table, chosen))

    st.markdown(f"**Confusion matrix at {chosen:.0%}**")
    st.dataframe(pd.DataFrame(
        [[int(row['tn']), int(row['Good applicants rejected'])],
         [int(row['Missed defaulters']), int(row['tp'])]],
        index=['Actually repaid', 'Actually defaulted'],
        columns=['Predicted repay', 'Predicted default'],
    ))

    st.markdown("**ROC curve**")
    st.caption("How well the model ranks defaulters above non-defaulters. The diagonal is random guessing.")
    st.altair_chart(roc_chart(y_test, test_prob))

    with st.expander("View threshold table"):
        st.dataframe(table[['threshold', 'Recall', 'Precision', 'Missed defaulters',
                            'Good applicants rejected', 'Cost']], hide_index=True)

# ----- Tab 3: what the data shows -----
with tab_data:
    st.caption(f"Default rates across all {len(df):,} cleaned loans. The overall default rate is {df['loan_status'].mean():.0%}.")

    lti = df.assign(loan_percent_income=pd.cut(
        df['loan_percent_income'], bins=[-0.01, 0.1, 0.2, 0.3, 0.4, 1.0],
        labels=['Under 10%', '10–20%', '20–30%', '30–40%', 'Over 40%']))

    sections = [
        ("By loan grade", df, 'loan_grade', list('ABCDEFG'), False),
        ("By loan-to-income ratio", lti, 'loan_percent_income',
         ['Under 10%', '10–20%', '20–30%', '30–40%', 'Over 40%'], False),
        ("By home ownership", df, 'person_home_ownership', None, True),
        ("By loan purpose", df, 'loan_intent', None, True),
    ]
    for title, data, column, order, horizontal in sections:
        st.markdown(f"**{title}**")
        chart, table_view = default_rate_chart(data, column, order, horizontal)
        st.altair_chart(chart)
        with st.expander("View as table"):
            st.dataframe(table_view, hide_index=True)
