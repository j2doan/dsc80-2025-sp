# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'NMAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'NMAR', 'MAR', 'NMAR', 'MCAR']



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    """
    pay = payments.copy()
    
    pay['date_of_birth'] = pd.to_datetime(pay['date_of_birth'], errors='coerce')
    pay['age'] = 2024 - pay['date_of_birth'].dt.year

    pay = pay.dropna(subset=['age'])

    age_missing = pay.loc[pay['credit_card_number'].isna(), 'age'].values
    age_not_missing = pay.loc[~pay['credit_card_number'].isna(), 'age'].values

    obsv_diff = abs(np.mean(age_missing) - np.mean(age_not_missing))

    simulated_diffs = []

    for _ in range(1000):
        shuffled = pay.copy()
        shuffled['age'] = np.random.permutation(shuffled['age'].values)
        age_missing_shuffle = shuffled.loc[shuffled['credit_card_number'].isna(), 'age'].values
        age_not_missing_shuffle = shuffled.loc[~shuffled['credit_card_number'].isna(), 'age'].values
        simulated_diff = abs(np.mean(age_missing_shuffle) - np.mean(age_not_missing_shuffle))
        simulated_diffs.append(simulated_diff)

    p_value = np.mean(simulated_diffs >= obsv_diff)
    result = 'R' if p_value < 0.05 else 'NR'
    """
    return [0.14, 'NR']


def second_round():
    """
    pay = payments.copy()
    
    pay['date_of_birth'] = pd.to_datetime(pay['date_of_birth'], errors='coerce')
    pay['age'] = 2024 - pay['date_of_birth'].dt.year

    pay = pay.dropna(subset=['age'])

    age_missing = pay.loc[pay['credit_card_number'].isna(), 'age'].values
    age_not_missing = pay.loc[~pay['credit_card_number'].isna(), 'age'].values

    obsv_diff = stats.ks_2samp(age_missing, age_not_missing).statistic

    simulated_diffs = []

    for _ in range(1000):
        shuffled = pay.copy()
        shuffled['age'] = np.random.permutation(shuffled['age'].values)
        age_missing_shuffle = shuffled.loc[shuffled['credit_card_number'].isna(), 'age'].values
        age_not_missing_shuffle = shuffled.loc[~shuffled['credit_card_number'].isna(), 'age'].values
        simulated_diff = stats.ks_2samp(age_missing_shuffle, age_not_missing_shuffle).statistic
        simulated_diffs.append(simulated_diff)

    p_value = np.mean(simulated_diffs >= obsv_diff)
    result = 'R' if p_value < 0.05 else 'NR'
    dependency = 'D' if result == 'R' else 'ND'
    """
    return [0.019, 'R', 'D']



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):

    child_X = heights.drop(columns=['child', 'father'])

    p_values = []

    for col_name in child_X.columns:
        
        df = heights[['father', col_name]]

        father_na = df.loc[df[col_name].isna(), 'father']
        father_no_na = df.loc[~df[col_name].isna(), 'father']

        ks_result = stats.ks_2samp(
            father_na, 
            father_no_na
        )

        p_value = ks_result.pvalue

        p_values.append(p_value)

    p_value_series = pd.Series(p_values, index=child_X.columns)

    return p_value_series


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    
    father_bins = pd.qcut(new_heights['father'], q=4)

    binned = new_heights.assign(father_bin=father_bins)

    imputed_values = (
        binned.groupby('father_bin')['child']
        .transform(lambda x: x.fillna(x.mean()))
    )

    return imputed_values


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    observed = child.dropna()

    counts, bin_edges = np.histogram(observed, bins=10, density=False)

    bin_widths = np.diff(bin_edges)

    bin_areas = counts * bin_widths
    probs = bin_areas / bin_areas.sum()

    chosen_bins = np.random.choice(len(probs), size=N, p=probs)

    lefts = bin_edges[chosen_bins]
    rights = bin_edges[chosen_bins + 1]

    samples = np.random.uniform(lefts, rights)

    return np.round(np.random.uniform(lefts, rights), 1)

def impute_height_quant(child):
    filled = child.copy()

    n_missing = filled.isna().sum()

    imputation_pool = quantitative_distribution(child, n_missing)

    filled.loc[filled.isna()] = imputation_pool

    return filled


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    return [[1, 2, 2, 1], ['https://www.scrapethissite.com/robots.txt', 'https://www.instagram.com/robots.txt']]
