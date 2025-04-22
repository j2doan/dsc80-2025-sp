# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    """
    data = [
        ['A', 'a', 0],
        ['B', 'b', 1],
        ['C', 'c', 2],
        ['D', 'd', 3],
        ['E', 'e', 4]
    ]
    tricky_1 = pd.DataFrame(data, columns=['Name', 'Name', 'Age'])
    tricky_1.to_csv("D:/UCSD/DSC80/dsc80-2025-sp/labs/lab02/tricky_1.csv", index=False)
    tricky_2 = pd.read_csv('D:/UCSD/DSC80/dsc80-2025-sp/labs/lab02/tricky_1.csv')

    if list(tricky_1.columns) == list(tricky_2.columns):
        return 2
    else:
        return 3
    """
    return 3

def trick_bool():
    """
    data = [
        ['A', 'a', 0, 1],
        ['B', 'b', 1, 2],
        ['C', 'c', 2, 3],
        ['D', 'd', 3, 4]
    ]
    bools = pd.DataFrame(data, columns=[True, True, False, False])
    return bools[True]
    return bools[[True, True, False, False]]
    return bools[[True, False]]
    """
    return [4, 10, 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    total_count = len(df)

    num_nonnull = df.notna().sum()
    prop_nonnull = num_nonnull / total_count

    num_distinct = df.nunique(dropna=True)
    prop_distinct = num_distinct / num_nonnull

    stats_df = pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })

    return stats_df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    new_df = pd.DataFrame()
    for col in df.columns:
        x = df.groupby(col).count()
        sort = x.sort_values(by=x.columns[0], ascending=False)
        val = sort.index[:N]
        count = sort[sort.columns[0]].tolist()[:N]
        new_df[f"{col}_values"] = val
        new_df[f"{col}_counts"] = count
    return new_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    most_powers = powers.set_index('hero_names').T.sum().sort_values(ascending=False).index[0]

    common_among_flight = powers[powers['Flight'] == True].drop(columns=['Flight']).set_index('hero_names').sum().sort_values(ascending=False).index[0]

    num_powers = powers.set_index('hero_names').T.sum().sort_values(ascending=False)
    solo_powers = num_powers[num_powers == 1].index.tolist()

    common_among_solo = powers[powers['hero_names'].isin(solo_powers)].set_index('hero_names').sum().sort_values(ascending=False).index[0]

    return [most_powers, common_among_flight, common_among_solo]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace('-', np.nan).replace(-99, np.nan) 


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    """
    clean_out[(clean_out['Race'] == 'Mutant') & (clean_out['Hair color'] == 'No Hair')].sort_values(by='Height', ascending=False)
    # Onslaught

    x = clean_out['Publisher'].value_counts()
    y = x[x > 5].index.tolist()
    z = clean_out[clean_out['Publisher'].isin(y)]
    w = z[['Publisher', 'Race']].groupby('Publisher').size().sort_index()
    v = z[z['Race'] == 'Human'][['Publisher', 'Race']].groupby('Publisher').size().sort_index()
    proportion = v / w
    proportion.sort_values(ascending=False).index[0]
    # George Lucas

    good = clean_out[clean_out['Alignment'] == 'good']['Height'].mean()
    bad = clean_out[clean_out['Alignment'] == 'bad']['Height'].mean()
    # bad

    u = clean_out[clean_out['Publisher'].isin(['Marvel Comics', 'DC Comics'])]
    total = u['Publisher'].value_counts()
    bad_counts = u[u['Alignment'] == 'bad']['Publisher'].value_counts()
    proportion_bad = bad_counts / total
    top_publisher = proportion_bad.sort_values(ascending=False).index[0]
    top_publisher
    # Marvel Comics

    publishers = clean_out[~clean_out['Publisher'].isin(['Marvel Comics', 'DC Comics'])].groupby('Publisher').count()
    publishers.sort_values(by=publishers.columns[0], ascending=False).index[0]
    # NBC - Heroes

    height_mean = clean_out['Height'].mean()
    height_std = clean_out['Height'].std()
    weight_mean = clean_out['Weight'].mean()
    weight_std = clean_out['Weight'].std()
    tall_and_light = clean_out[(clean_out['Height'] > height_mean + height_std) & (clean_out['Weight'] < weight_mean - weight_std)]
    tall_and_light['name'].iloc[0]
    # Groot
    """
    return ['Onslaught', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    new_df = df.copy()
    new_df['institution'] = new_df['institution'].str.replace('\n', ', ', regex=True)

    new_df['broad_impact'] = pd.to_numeric(new_df['broad_impact'], errors='raise').astype(int) 


    new_df[['nation', 'national_rank_cleaned']] = df['national_rank'].str.split(', ', expand=True)
    new_df['national_rank_cleaned'] = new_df['national_rank_cleaned'].astype(int)
    new_df['nation'] = new_df['nation'].replace({
        'USA': 'United States',
        'UK': 'United Kingdom',
        'Czechia': 'Czech Republic'
    })

    new_df['is_r1_public'] = ((new_df['control'] == 'Public') & new_df[['control', 'city', 'state']].notna().all(axis=1))
    return new_df

def university_info(cleaned):
    state_counts = cleaned['state'].value_counts()
    three_plus = state_counts[state_counts >= 3].index
    mean_scores = cleaned[cleaned['state'].isin(three_plus)].groupby('state')['score'].mean()
    lowest_state = mean_scores.idxmin()
    # AL

    national = cleaned[cleaned['world_rank'] <= 100]
    faculty = cleaned[(cleaned['national_rank_cleaned'] <= 100) & (cleaned['quality_of_faculty'] <= 100)]
    proportion = len(faculty) / len(national)
    # 0.71

    names = cleaned.groupby('state')['is_r1_public']
    private_proportion = (names.apply(lambda x: (x == False).sum() / len(x)))
    greater_than_fifty_percent = (private_proportion >= 0.5).sum()
    # 13

    worst_world_best_nation = cleaned[cleaned['national_rank_cleaned'] == 1][['world_rank', 'institution']].sort_values(by='world_rank', ascending=False).iloc[0]['institution']
    # University of Bucharest

    return [lowest_state, proportion, greater_than_fifty_percent, worst_world_best_nation]

