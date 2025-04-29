# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    new_loans = loans.copy()
    new_loans['issue_d'] = new_loans['issue_d'].apply(lambda x: pd.Timestamp(x))

    new_loans['term'] = new_loans['term'].str.replace(' months', '').astype(int)

    new_loans['emp_title'] = new_loans['emp_title'].str.lower().str.strip().apply(lambda x: 'registered nurse' if x == 'rn' else x)

    new_loans['term_end'] = new_loans.apply(lambda row: row['issue_d'] + pd.DateOffset(months=row['term']), axis=1)
    return new_loans


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    output = {}
    for col1, col2 in pairs:
        title = '_'.join(['r', col1, col2])
        r = df[col1].corr(df[col2])
        output[title] = r

    return pd.Series(output)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    
    def set_fico_bounds(fico):
        if fico >= 580 and fico < 670:
            return '[580, 670)'
        elif fico >= 670 and fico < 740:
            return '[670, 740)'
        elif fico >= 740 and fico < 800:
            return '[740, 800)'
        elif fico >= 800 and fico < 850:
            return '[800, 850)'
        else:
            return np.nan
    
    loans_copy = loans.copy()

    loans_copy['Credit Score Range'] = loans_copy['fico_range_low'].apply(set_fico_bounds)

    loans_copy = loans_copy.dropna(subset=['Credit Score Range'])

    credit_score_order = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']
    term_order = [36, 60]

    fig = px.box(
        loans_copy, 
        x='Credit Score Range', 
        y='int_rate', 
        color='term', 
        color_discrete_map={36: 'purple', 60: 'gold'},
        title="Interest Rate vs. Credit Score",
        labels={'Credit Score Range': 'Credit Score Range', 'int_rate': 'Interest Rate (%)', 'term': 'Loan Length (Months)'},
        category_orders={
            'Credit Score Range': credit_score_order,
            'term': term_order
        }
    )

    fig.update_traces(quartilemethod="exclusive")  # Using exclusive quartile method

    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    with_statement = loans[loans['desc'].notna() == True]['int_rate']
    without_statement = loans[loans['desc'].notna() == False]['int_rate']

    obsv_stat = with_statement.mean() - without_statement.mean()

    all_int_rates = np.concatenate([with_statement, without_statement])

    perm_diffs = []

    for _ in range(N):
        np.random.shuffle(all_int_rates)

        perm_with = all_int_rates[:len(with_statement)]
        perm_without = all_int_rates[len(with_statement):]

        perm_stat = perm_with.mean() - perm_without.mean()

        perm_diffs.append(perm_stat)

    p_val = np.mean(np.array(perm_diffs) >= obsv_stat)

    return p_val
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    '''
    Put your justification here in this multi-line string.
    Make sure to return your string!
    '''
    return 'If we just consider the desc column by itself, we can argue ' \
    'NMAR based on a reasoning where people with lower credit scores would' \
    'more likely give a personal statement to boost their chances of getting' \
    'a loan, making the missing values dependent on the values themselves.'

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    total = 0
    curr_multiplier = 0
    prev_threshold = 0
    for i, (multiplier, threshold) in enumerate(brackets):
        if i == 0:
            if income > threshold:
                curr_multiplier = multiplier
                prev_threshold = threshold
                continue
        else:
            if income > (threshold - prev_threshold):
                total += curr_multiplier * (threshold - prev_threshold)
                income -= (threshold - prev_threshold)
                prev_threshold = threshold
                curr_multiplier = multiplier
            else:
                break

    total += curr_multiplier * income

    return total


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    ...
    
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    # Now it's your turn:
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    ...
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': [..., ...],
        'quantitative_column': ...,
        'categorical_column': ...
    }
