"""
permits_data.py — loads and cleans the SF building permits data for every model.

All model notebooks import from this file, so every model uses the same
rows, features, train/test split, preprocessing and CV folds.

Functions:
    load_permits()      : downloads the raw permits file.
    resolved_permits()  : keeps permits with a final outcome, filed 2000–2013,
                          and adds the target y (1 = did not finish, 0 = completed).
    add_features()      : builds the features known on the day of filing
                          (log cost, unit/story changes, filing year, ...).
    batch_id()          : labels permits from the same lot and filing day as one project.
    make_preprocessor() : recipe to fill missing values, scale numbers and
                          one-hot encode categories (unfitted; fitted inside each model).
    make_cv()           : grouped 5-fold cross-validation with a fixed seed, so every
                          model is tuned on the same folds.
    get_permit_data()   : runs all of the above and returns
                          X_train, X_test, y_train, y_test, groups_train, preprocessor.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# SF Permits data 
PERMITS_URL = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
               "main/data/2022/2022-07-05/sf_permits.csv")

# Set random seed for reproducibility/ consistency
RANDOM_STATE = 42

# Only permits filed in these years are used.
# Why: permits filed after 2013 had not all had time to finish or expire when
# the data was collected, so many of them would still be open (no outcome yet).
YEAR_MIN, YEAR_MAX = 2000, 2013

# The target. These statuses are final outcomes.
# 0 = completed, 1 = did not finish (expired, cancelled, withdrawn, ...).
# Why 1 = "did not finish": that is the outcome the city wants to flag.
FINAL_STATUS = {
    'complete': 0,
    'expired': 1, 'cancelled': 1, 'withdrawn': 1,
    'disapproved': 1, 'revoked': 1, 'denied': 1,
}

# The features the models use. Everything else in the raw file is dropped.
# Why: the raw file also has columns recorded AFTER filing (completed date,
# issued date, revised cost, status...). Those would give the answer away
# ("leakage"), and the city would not have them when a permit is filed.
# ID columns (permit number, street name, ...) are dropped because they are
# labels, not information about the project.
NUMERIC = [
    'log_estimated_cost',          # estimated cost, on a log scale
    'cost_is_placeholder',         # 1 if cost was entered as $0 or $1
    'plansets',                    # number of plan sets submitted
    'existing_units', 'proposed_units', 'unit_change',
    'number_of_existing_stories', 'number_of_proposed_stories', 'story_change',
    'filing_year', 'filing_month',
]

CATEGORICAL = [
    'permit_type_definition',      # e.g. "otg alterations permit"
    'existing_use', 'proposed_use', 'use_changes',
    'existing_construction_type_description', 'proposed_construction_type_description',
    'structural_notification', 'site_permit', 'fire_only_permit',
    'voluntary_soft_story_retrofit',
    'neighborhoods_analysis_boundaries',   # neighborhood
    'supervisor_district',
]

# Load the raw file
def load_permits(url=PERMITS_URL):
    """Download the raw permits file (about 200,000 rows)."""
    return pd.read_csv(url, low_memory=False)

# Keep resolved permits and create the target y
def resolved_permits(raw):
    """Keep permits with a final outcome, filed 2000–2013, and add y."""
    # Keep only final outcomes. Why: a permit that is still open has no
    # answer yet, so we cannot train or test on it.
    df = raw[raw['status'].isin(FINAL_STATUS)].copy()

    # Keep filing years 2000–2013 (see YEAR_MIN / YEAR_MAX above).
    df['filed'] = pd.to_datetime(df['filed_date'])
    df = df[df['filed'].dt.year.between(YEAR_MIN, YEAR_MAX)]

    # Target: 1 = did not finish, 0 = completed.
    df['y'] = df['status'].map(FINAL_STATUS)
    return df

# Build features known on the day of filing
def add_features(df):
    """Create the model features. All of them are known when the permit is filed."""
    out = df.copy()
    cost = out['estimated_cost']

    # Some applicants enter $0 or $1 as a placeholder. Flag it, and treat
    # the cost itself as missing so it does not look like a real $1 project.
    out['cost_is_placeholder'] = (cost <= 1).astype(int)

    # Cost ranges from $1 to over $500 million. A log scale stops a few huge
    # projects from dominating (each step = 10x the cost).
    out['log_estimated_cost'] = np.log10(cost.where(cost > 1))

    # How much the project changes the building.
    out['unit_change'] = out['proposed_units'] - out['existing_units']
    out['story_change'] = out['number_of_proposed_stories'] - out['number_of_existing_stories']

    # Does the building's use change (e.g. office -> apartments)?
    # Left blank if either use is unknown.
    both_known = out['existing_use'].notna() & out['proposed_use'].notna()
    changes = out['existing_use'] != out['proposed_use']
    out['use_changes'] = changes.map({True: 'yes', False: 'no'}).where(both_known)

    # When the permit was filed.
    out['filing_year'] = out['filed'].dt.year
    out['filing_month'] = out['filed'].dt.month

    # Supervisor district is a label (1–11), not a quantity, so store it as text.
    out['supervisor_district'] = (out['supervisor_district']
                                  .astype('Int64').astype(str).replace('<NA>', np.nan))

    # For these yes/no flags, a blank means "not flagged", so fill with 'N'.
    for col in ['structural_notification', 'site_permit',
                'fire_only_permit', 'voluntary_soft_story_retrofit']:
        out[col] = out[col].fillna('N')

    return out

# Helper function: group permits that belong to the same project
def batch_id(df):
    """Permits filed for the same lot on the same day are treated as one project."""
    # Why: one project often files several permits on the same day. If some
    # went to train and some to test, the model would be tested on projects
    # it had already seen, and the scores would look better than they are.
    return (df['block'].astype(str) + '-' + df['lot'].astype(str)
            + '-' + df['filed_date'].astype(str))

# Preprocessor (fill missing values, scale, one-hot encode)
def make_preprocessor():
    """Returns an unfitted preprocessor to put at the start of a model Pipeline."""
    # Why unfitted: inside a Pipeline it learns medians and scales from the
    # training folds only, so no information leaks in from the test data.

    # Numbers: fill blanks with the median, add a "was missing" flag column,
    # then standardize (mean 0, sd 1). Scaling matters for logistic regression
    # and the neural network; trees ignore it.
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])

    # Categories: fill blanks with the word 'missing', then one-hot encode.
    # Categories with fewer than 50 permits are pooled into one "infrequent"
    # column, so rare labels don't create hundreds of near-empty columns.
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50)),
    ])

    return ColumnTransformer([
        ('num', numeric, NUMERIC),
        ('cat', categorical, CATEGORICAL),
    ])

# Cross-validation
def make_cv():
    """5-fold cross-validation that keeps each project inside one fold."""
    # Stratified: every fold has about the same share of unfinished permits.
    # Grouped: a project's permits never sit in both the training and validation fold.
    # Use it as: GridSearchCV(..., cv=make_cv()).fit(X_train, y_train, groups=groups_train)
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Final function 
def get_permit_data(raw=None, test_size=0.2):
    """
    Runs Steps 1–6 and returns:
        X_train, X_test, y_train, y_test, groups_train, preprocessor
    """
    # Steps 1–3: load, filter, add features.
    if raw is None:
        raw = load_permits()
    df = add_features(resolved_permits(raw)).reset_index(drop=True)

    # Step 4: keep only the model features. This is where the leakage and
    # ID columns get dropped.
    X = df[NUMERIC + CATEGORICAL]
    y = df['y']
    groups = batch_id(df)

    # Step 5: 80/20 train/test split, keeping each project on one side.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    # Step 6: the preprocessor, still unfitted.
    preprocessor = make_preprocessor()

    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx], preprocessor)
