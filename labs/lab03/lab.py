# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):

    if not Path(dirname).exists() or not Path(dirname).is_dir():
        raise FileNotFoundError()
    
    files = list(Path(dirname).iterdir())
    survey_files = [f for f in files if f.name.startswith("survey") and f.suffix == '.csv']

    dfs = []

    expected_cols = ['first name', 'last name', 'current company', 'job title', 'email', 'university']

    for file in survey_files:

        df = pd.read_csv(file)

        df.columns = df.columns.str.lower().str.strip()

        col_mapping = {
                'first': 'first name',
                'first_name': 'first name',
                'lastname': 'last name',
                'last': 'last name',
                'last_name': 'last name',
                'company': 'current company',
                'currentcompany': 'current company',
                'title': 'job title',
                'jobtitle': 'job title',
                'job': 'job title',
                'email address': 'email',
                'email': 'email',
                'school': 'university',
                'university': 'university'
            }

        df.rename(columns=col_mapping, inplace=True)

        df = df[[col for col in expected_cols if col in df.columns]]

        df = df.reindex(columns=expected_cols)

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    return result



def com_stats(df):
    numerator = df[(df['job title'].str.contains('Programmer', na=False)) & (df['university'].str.contains('Ohio', na=False))].shape[0]
    denominator = df[df['university'].str.contains('Ohio', na=False)].shape[0]
    one = numerator/denominator

    filtered_jobs = df['job title'].dropna()
    engineer_jobs = filtered_jobs[filtered_jobs.str.endswith('Engineer')]
    two = engineer_jobs.nunique()

    three = df['job title'].dropna().loc[df['job title'].dropna().str.len().idxmax()]

    four = df['job title'].dropna()[df['job title'].dropna().str.lower().str.contains('manager')].shape[0]

    return [one, two, three, four]



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    files = list(Path(dirname).iterdir())

    df = pd.DataFrame()

    for file in files:
        curr_df = pd.read_csv(file)

        # since each survey answer is in the 2nd column
        df = pd.concat([df, curr_df.iloc[:, 1]], axis=1)

    # the files are already sorted by id, so this will match
    df['id'] = np.arange(1, 1001)

    df = df.set_index('id')

    return df


def check_credit(df):
    df_clean = df.copy()
    df_clean = df_clean.replace('(no genres listed)', np.nan)
    df_clean = df_clean.fillna(0)

    total_check = df_clean.copy().drop(columns='name')
    total_check[total_check != 0] = 1

    # calc individual scores
    individual_scores = total_check.sum(axis=1)

    new_individual_scores = individual_scores.apply(lambda x: 5 if int(x) >= 3 else 0)

    total_check['ec'] = new_individual_scores

    # calc group bonus scores
    group_scores = total_check[['movie', 'genre', 'animal', 'plant', 'color']].sum(axis=0)

    bonus_ec = min(group_scores[group_scores.astype(int) > 900].count(), 2)

    # add it up
    total_check['ec'] += bonus_ec

    total_check['name'] = df_clean['name']

    return total_check[['name', 'ec']]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    df = pets.merge(procedure_history, on='PetID')
    return df.groupby('ProcedureType').count().idxmax().iloc[0]

def pet_name_by_owner(owners, pets):
    df2 = owners.merge(pets, on='OwnerID')
    grouped = df2.groupby(['OwnerID', 'Name_x'])['Name_y'].agg(list).reset_index()
    grouped['Name_y'] = grouped['Name_y'].apply(lambda x: x[0] if len(x) == 1 else x)
    return grouped[['Name_x', 'Name_y']].set_index('Name_x')


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    merged_all = owners.merge(pets, on='OwnerID', how='left').merge(pd.merge(procedure_history, procedure_detail, on=['ProcedureSubCode', 'ProcedureType']), on='PetID', how='left')
    return merged_all.groupby('City')['Price'].sum()

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    return sales.groupby('Name')[['Total']].mean().rename(columns={'Total':'Average Sales'})

def product_name(sales):
    return sales.pivot_table(index='Name', columns='Product', values='Total', aggfunc='sum')

def count_product(sales):
    return sales.pivot_table(index=['Product', 'Name'], columns='Date', values='Total', aggfunc='count').fillna(0).astype(int)

def total_by_month(sales):
    pivot_sales = sales.copy()

    month_map = {
        '01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
    }

    pivot_sales['Month'] = pivot_sales['Date'].str[:2]

    pivot_sales['Month'] = pivot_sales['Month'].map(month_map)

    return pivot_sales.pivot_table(index=['Name', 'Product'], columns='Month', values='Total', aggfunc='sum').fillna(0).astype(int)