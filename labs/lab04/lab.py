# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    new_login = login.copy()
    new_login['Time'] = new_login['Time'].apply(pd.Timestamp)
    new_login['Time'] = new_login['Time'].apply(lambda x: x if 20 > x.hour >= 16 else np.nan)
    new_login['Time'] = new_login['Time'].dropna()
    return new_login.groupby('Login Id').count()


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    current_date = pd.Timestamp('2024-01-31 23:59:00')

    login['Time'] = pd.to_datetime(login['Time'])
        
    result = login.groupby('Login Id').agg(total_logins=('Time', 'count'), first_login=('Time', 'min'))
        
    result['days_active'] = (current_date - result['first_login']).dt.days
        
    result['logins_per_day'] = result['total_logins'] / result['days_active']
        
    return result['logins_per_day']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):
    obsv = np.array([0.94, 0.06])
    expect = np.array([0.96, 0.04]) # 235 good, 15 burnt
    N = 10000
    samples = np.random.multinomial(250, expect, size=N) / 250
    tvds = np.abs(samples - expect).sum(axis=1) / 2

    obsv_tvd = np.abs(obsv - expect).sum() / 2

    p_val = (tvds >= obsv_tvd).mean()
    
    return float(p_val)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1]
    
def bhbe_col(heroes):
    heroes['blond and blue eyes'] = (heroes['Eye color'].str.lower().str.contains('blue')) & (heroes['Hair color'].str.lower().str.contains('blond'))
    return heroes['blond and blue eyes']

def superheroes_observed_statistic(heroes):
    x = bhbe_col(heroes)
    counts = heroes[x == True]['Alignment'].value_counts()
    y = counts / sum(counts)
    return y['good']

def simulate_bhbe_null(heroes, N):
    x = bhbe_col(heroes)
    sample_size = heroes[x == True]['Alignment'].size

    aligns = heroes['Alignment'].to_numpy()
    is_good = (aligns == 'good').astype(int)
    samples = np.random.choice(is_good, size=(N, sample_size), replace=True)
    samples = samples.mean(axis=1)

    return samples

def superheroes_p_value(heroes):
    obsv = superheroes_observed_statistic(heroes) # obsv = proportion of goods in BLUE/BLOND
    expect = (heroes['Alignment'] == 'good').mean() # expected = POPULATION proportion of goods
    samples = simulate_bhbe_null(heroes, 100000) # sample = many proportions of obsv sample size, taken from POPULATION

    tvds = np.abs(samples - expect) / 2

    obsv_tvd = np.abs(obsv - expect) / 2

    p_val = (tvds >= obsv_tvd).mean()

    p_val = float(p_val)

    if p_val < 0.01:
        return [p_val, 'Reject']
    else:
        return [p_val, 'Fail to reject']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    orange_mean = data[[col, 'Factory']].groupby('Factory').mean()
    waco = orange_mean.loc['Waco']
    york = orange_mean.loc['Yorkville']

    return np.abs(waco - york).iloc[0]


def simulate_null(data, col='orange'):
    with_shuffled = data.copy()
    with_shuffled['Factory'] = np.random.permutation(with_shuffled['Factory'])

    return diff_of_means(with_shuffled, col)


def color_p_value(data, col='orange'):
    observed_diff = diff_of_means(data, col)
    simulated_diffs = np.array([simulate_null(data, col) for _ in range(1000)])
    p_value = np.mean(simulated_diffs >= observed_diff)
    
    return p_value


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [('yellow', np.round(0.0, 3)), ('orange', np.round(0.049, 3)), 
            ('red', np.round(0.235, 3)), ('green', np.round(0.468, 3)), 
            ('purple', np.round(0.966, 3))]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.009, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
