import pandas
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier


# Establish your constants 
DATA_PATH = 'data/credit_risk_dataset.csv'
MODEL_PATH = 'models/credit_model.joblib'
TARGET = 'loan_status'
RANDOM_STATE = 77
TEST_SIZE = 0.2 
THRESHOLD = 0.2
FN_COST = 7
FP_COST = 1

NUMERIC_COLUMNS = [
    'person_age', 
    'person_income', 
    'person_emp_length', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_cred_hist_length'
]
CATEGORICAL_COLUMNS = [
    'person_home_ownership', 
    'cb_person_default_on_file', 
    'loan_intent', 
    'loan_grade'
]

def load_data(path):
    df = pandas.read_csv(path)
    return df

def clean_data(df):
    # Drop duplicated rows
    df = df.drop_duplicates()
    # Drop Impossible ages
    df = df[df['person_age'] < 85] 
    # Drop Impossible employment lengths
    df = df[(df['person_emp_length'] < 70) | (df['person_emp_length'].isna())]

    return df

def split_data(df):
    # x will be every column except target
    x = df.drop(columns=[TARGET])
    # y will be the target column
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return x_train, x_test, y_train, y_test

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )

    return preprocessor

def build_model():
    model = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    return model

# Grid search over C, returns the best pipeline
def tune_model(x_train, y_train):
    params = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(build_model(), param_grid=params, cv=5,
                               scoring='roc_auc', n_jobs=1)
    grid_search.fit(x_train, y_train)

    print(f"Best C: {grid_search.best_params_['classifier__C']}")
    print(f"Best cross-validated AUC: {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_


# Final evaluation on the test set at the chosen threshold
def evaluate_model(model, x_test, y_test, threshold):
    y_prob = model.predict_proba(x_test)[:, 1]
    print(f"Test AUC: {roc_auc_score(y_test, y_prob):.3f}")

    y_pred = (y_prob >= threshold).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred


# Cost table across thresholds, using cross-validated training predictions
def compare_thresholds(model, x_train, y_train):
    train_prob = cross_val_predict(model, x_train, y_train, cv=5,
                                   method='predict_proba')[:, 1]

    print("\nTesting different threshold values:")
    print("-" * 130)
    for threshold in np.arange(0.10, 0.75, 0.05):
        pred = (train_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        cost = (FN_COST * fn) + (FP_COST * fp)
        print(f"Threshold: {threshold:>.2f} | Recall: {recall:.2f} | Precision: {precision:.2f}| False Negatives: {fn:>11} | False Positives: {fp:>18} | Cost: {cost:>5}")

# Print a table to evaluate features
def show_coefficients(model):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pandas.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'odds_ratio': np.exp(coefficients)
    })

    coef_df = coef_df.sort_values('coefficient', ascending=False)
    print("Coefficients sorted by effect on default risk:")
    print(coef_df.to_string(index=False))

    return coef_df

# Trains a dummy model w no features
def baseline_check(x_train, x_test, y_train, y_test):
    # this model will ignore all features and predict the majority class
    dummy = DummyClassifier(strategy='prior')
    dummy.fit(x_train, y_train)

    dummy_prob = dummy.predict_proba(x_test)[:, 1]

    print("\n=== Baseline check ===")
    print(f"Dummy model AUC: {roc_auc_score(y_test, dummy_prob):.3f}") 
    return dummy

def save_model(model, threshold, path):
    # Create the models/ folder if it doesn't exist yet.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Bundle the model and threshold in a dictionary so they're saved (and loaded) together
    bundle = {
        'model': model,
        'threshold': threshold,
    }

    # Write the bundle to a file. It now survives after the script ends.
    joblib.dump(bundle, path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    df = clean_data(load_data(DATA_PATH))
    x_train, x_test, y_train, y_test = split_data(df)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    best_model = tune_model(x_train, y_train)
    evaluate_model(best_model, x_test, y_test, THRESHOLD)
    compare_thresholds(best_model, x_train, y_train)

    print("\n=== Interpretation Checks ===")
    show_coefficients(best_model)
    baseline_check(x_train, x_test, y_train, y_test)

    save_model(best_model, THRESHOLD, MODEL_PATH)

