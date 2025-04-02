import numpy as np
import pandas as pd
from datasets import load_dataset
from numpy import random as r
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import textwrap

sex = {1: 'Male', 2: 'Female'}
df1 = pd.DataFrame({'index': [1, 2], 'sex': ['Male', 'Female']})
race = {1: 'White', 2: 'Black', 4: 'Asian'}
origin = {1: 'No Hispanic Origin', 2: 'Hispanic Origin'}

dataset = load_dataset("sacrificialpancakes/synthetic_demographics_seed")['train'].to_pandas()

dd = pd.DataFrame(dataset)

columns_names = dd.columns[::]

random_sample = dd.sample(5, random_state=10)

#1. All info about dataset  

def info_about_dataset(dataframe, column_name_start=0, column_name_end=20, column_num_start=0, column_num_end=100000):
    start_time  = time.time()

    cols = dataframe.columns[column_name_start : column_name_end : ].tolist()
    null_values_in_columns = [col for col in cols if dataframe[col].iloc[column_num_start : column_num_end].isnull().sum() > 0]
    sample = dataframe.sample(5, random_state=np.random.randint(1000))
    print(f"Dataset length is {dataframe.__len__()} rows")
    print(f"Dataset columns are: {cols}")
    print(f"Next columns have null values - {null_values_in_columns}")
    print(sample)

    end_time  = time.time()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

info_about_dataset(dd)

#2 Top 10 most popular first names

def get_top_popular_name(dataframe, column_name, amount):
    top_rows = dataframe[column_name].value_counts().head(amount)
    print(top_rows)
    plt.figure(figsize=(10, 5))
    plt.barh(top_rows.index, top_rows.values, height=0.5)
    plt.gca().invert_yaxis()
    plt.title(f'Top {amount} most popular {column_name.replace("_", " ").title()}\'s')
    plt.show()

get_top_popular_name(dd, 'last_name', 10)

def get_top_popular_name_by(dataframe, column_name_by, column_name_by_num, column_name, amount):
    top_first_name_by_race = (
        dataframe.groupby(by=[column_name_by, column_name])[column_name]
        .count()
        .reset_index(name='count')
        .query(f"{column_name_by} == {column_name_by_num}")
        .sort_values(by='count', ascending=False)
        .head(amount))

    print(top_first_name_by_race)
    
    plt.figure(figsize=(10, 5))
    plt.barh(top_first_name_by_race[column_name], top_first_name_by_race['count'], height=0.5)
    plt.gca().invert_yaxis()
    plt.title(f'Top {amount} most popular {column_name.replace("_", " ").title()}\'s where {column_name_by} is {globals().get(column_name_by)[column_name_by_num]}')
    plt.show()

get_top_popular_name_by(dd, 'sex', 2, 'first_name', 10)


#2	Analyze age distribution across different education levels and employment status.
def dist_by_education(dataframe):

    education_types = dataframe['education'].unique()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6)) 
    axes = axes.flatten()  

    for i,education in enumerate(education_types):
        axes[i].set_title(f'{education}')
        axes[i].hist(dataframe[dataframe['education'] == education]['age'], bins=20, alpha=0.5, label=education, color='blue', edgecolor="black")

    plt.suptitle('Age distribution')
    plt.tight_layout()
    plt.show()

dist_by_education(dd)

# Find the youngest and oldest people in the labor force.
def youngest_oldest_employee_by_education(dataframe, education_type):
    data = dataframe['age'].where((dataframe['is_employed']) & (dataframe['education'] == education_type))
    youngest = data.min().__round__()
    oldest = data.max().__round__()
    print(f"The youngest person in the labor force is {youngest} years old")
    print(f"The oldest person in the labor force is {oldest} years old")

youngest_oldest_employee_by_education(dd, 'Doctorate')

# Associate's Degree
# High School
# Bachelor's Degree
# No High School
# Doctorate
# Some College
# Master's Degree
def groping_by_age(dataframe):
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']

    dataframe['age_group'] = pd.cut(dataframe['age'], bins=bins, labels=labels, right=True)

    age_distribution = dataframe.groupby('age_group', observed=False)['age'].count()

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))

    plt.title('Age distribution')
    plt.barh(age_distribution.index, age_distribution.values, height=0.5)
    plt.gca().invert_yaxis()
    plt.show()

groping_by_age(dd)

# Explore employment rates for different age groups and states.
def groping_by_state_and_age(dataframe):

    top_states = dataframe['state'].value_counts().head(6).keys()

    fig, ax = plt.subplots(2, 3, figsize=(12, 6)) 
    ax = ax.flatten()  

    for i, state in enumerate(top_states):
        ax[i].set_title(f'{state}')
        ax[i].hist(dataframe[dataframe['state'] == state]['age'], bins=20, alpha=0.5, color='blue', edgecolor="black")
        ax[i].set_ylim(0, 30000)
        ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))

    plt.suptitle('Age distribution by top states', fontsize=16)
    plt.tight_layout()
    plt.show()


# 3️⃣ Birthdate & Generational Analysis
# •	Convert birthdate to year of birth and classify individuals into generations (e.g., Gen Z, Millennials).

def birthdate_to_year(dataframe):
    dataframe['birthdate'] = pd.to_datetime(dataframe['birthdate'], errors='coerce')
    dataframe['birth_year'] = dataframe['birthdate'].dt.year
    dataframe['birth_year'] = dataframe['birth_year'].fillna(0).astype(int)

    bins = [1928, 1946, 1965, 1981, 1997, 2013, 2025]
    labels = ['The Silent Generation', 'Baby Boomers', 'Generation X', 'Millennials (Gen Y)', 'Generation Z', 'Generation Alpha']

    dataframe['generation'] = pd.cut(dataframe['birth_year'], bins = bins, labels=labels, right=True)

    result = dataframe.groupby('generation', observed=False)['generation'].count().sort_values(ascending=False)
    
    print(result)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))
    plt.barh(result.index, result.values, color='blue', edgecolor="black")
    # plt.yticks(rotation=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

birthdate_to_year(dd)

# •	Compare employment rates across generations.

def employment_by_generation(dataframe):
    dataframe['birthdate'] = pd.to_datetime(dataframe['birthdate'], errors='coerce')
    dataframe['birth_year'] = dataframe['birthdate'].dt.year
    dataframe['birth_year'] = dataframe['birth_year'].fillna(0).astype(int)

    bins = [1928, 1946, 1965, 1981, 1997, 2013, 2025]
    labels = ['The Silent Generation', 'Baby Boomers', 'Generation X', 'Millennials (Gen Y)', 'Generation Z', 'Generation Alpha']

    dataframe['generation'] = pd.cut(dataframe['birth_year'], bins = bins, labels=labels, right=True)
    result = dataframe.groupby(by='generation', observed=False)['is_employed'].mean().sort_values(ascending=False)
    print(result)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{np.round(x*100, 2)}%'))
    plt.xlim(0.60, 0.605)
    plt.barh(result.index, result.values, color='blue', edgecolor="black")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

employment_by_generation(dd)



# 4️⃣ Gender & Employment Disparities
# •	Compare employment rates between sex categories.

def employment_by_sex(dataframe):
    result = dataframe.groupby(by='sex', observed=False)['is_employed'].mean().sort_values(ascending=False)

    print(result)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_yticks(range(len(df1.sex)))
    ax.set_yticklabels(df1.sex) 
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{np.round(x*100, 2)}%'))
    plt.xlim(0.60, 0.605)
    plt.barh(range(len(df1.sex)), result.values, color='blue', edgecolor="black")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

employment_by_sex(dd)


# •	Analyze if occupation categories differ by gender.

def occ_by_gender(dataframe):  
    fig, ax = plt.subplots(2, 1, figsize=(9, 5))

    for i in range(1,3):
        result = (dataframe.loc[dataframe['sex'] == i]  
          .groupby(['sex', 'occupation_category'], observed=False)
          .size()  
          .sort_values(ascending=False)
          .head(10))
        
        # print(result)
        ax[i-1].set_title(f'{sex[i]}')
        ax[i-1].barh(result.index.get_level_values(1), result.values, color='blue', edgecolor="black")
        ax[i-1].tick_params(axis='y', labelsize=10)
        ax[i-1].invert_yaxis()  
        
    
    plt.suptitle('Top 10 occupation categories by gender', fontsize=18)
    plt.tight_layout()
    plt.show()


occ_by_gender(dd)


# 5️⃣ Racial & Ethnic Disparities in Employment
# •	Analyze unemployment rates by race & origin.

def unemp_rate_by_race_and_origin(dataframe, race, origin):
    result = dataframe.groupby


# •	Check if people with higher education levels tend to work in specific occupations.

def occupation_by_education(dataframe, is_include_unemployed=True):  

    education_types = dataframe['education'].unique()

    for i,education in enumerate(education_types):
        plt.title(f'{education}')
        top_ocupations = dataframe[(dataframe['education'] == education) & 
                           ((dataframe['occupation_category'] != 'Unemployed') | is_include_unemployed)]['occupation_category'].value_counts().head(5)
        wrapped_labels = [textwrap.fill(label, width=20) for label in top_ocupations.index]
        plt.barh(wrapped_labels, top_ocupations.values, color='blue', edgecolor="black")
        plt.gca().invert_yaxis()  
        plt.tight_layout()
        plt.show()


occupation_by_education(dd, True)


# •	Compare education levels across races/origins.

def edd_by_race(dataframe, parameter_object):

    fig, axes = plt.subplots(3, 1, figsize=(12, 6)) 
    axes = axes.flatten()  
    
    for i,param in enumerate(parameter_object):
        axes[i].set_title(f'{parameter_object[param]}')
        result = dataframe[dataframe['race'] == param]['education'].value_counts().sort_values(ascending=False)
        axes[i].barh(result.index, result.values, color='blue', edgecolor="black")
        axes[i].invert_yaxis()

    plt.suptitle('Compare education levels across races')
    plt.tight_layout()
    plt.show()

edd_by_race(dd, race)




# •	Determine which education levels are most common for students vs. non-students.

# •	Identify which demographics (age, sex, race, education) are most/least likely to be in the labor force.

# The most popular occupations for students and non-students

# •	Identify if certain states have a higher percentage of students.


#  Advanced: Predict Employment Status
# •	Build a predictive model to classify employment status based on: 
# o	Age, education, race, sex, state, is_student, is_in_labor_force
# •	Identify which features are the strongest predictors of employment.
