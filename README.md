# Data Analysis with Python

## Overview
This repository contains Python scripts for analyzing synthetic demographic data using **Pandas, NumPy, Seaborn, and Matplotlib**. The dataset includes information on race, gender, education, employment, and more.

## Dataset
The script loads a dataset from Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("sacrificialpancakes/synthetic_demographics_seed")["train"].to_pandas()
```

---
## Functions and Analysis

### 1. Dataset Overview
**Function:** `info_about_dataset(dataframe, column_name_start=0, column_name_end=20, column_num_start=0, column_num_end=100000)`
- Displays dataset size and columns
- Identifies missing values
- Shows a random sample of the data

### 2. Name Popularity Analysis
**Function:** `get_top_popular_name(dataframe, column_name, amount)`
- Identifies the most common names
- Displays a bar chart of name frequency

**Function:** `get_top_popular_name_by(dataframe, column_name_by, column_name_by_num, column_name, amount)`
- Finds the most common names filtered by race or gender

### 3. Age Distribution by Education
**Function:** `dist_by_education(dataframe)`
- Analyzes the distribution of ages within different education levels
- Generates histograms per education category

### 4. Employment Analysis
**Function:** `youngest_oldest_employee_by_education(dataframe, education_type)`
- Finds the youngest and oldest employed individuals within an education category

**Function:** `groping_by_age(dataframe)`
- Groups individuals by age ranges
- Displays a bar chart of age group distribution

**Function:** `groping_by_state_and_age(dataframe)`
- Compares age distributions in different states

### 5. Generational Analysis
**Function:** `birthdate_to_year(dataframe)`
- Converts birthdate to year of birth
- Classifies individuals into generations (Gen Z, Millennials, etc.)

**Function:** `employment_by_generation(dataframe)`
- Compares employment rates across generations

### 6. Gender & Employment Disparities
**Function:** `employment_by_sex(dataframe)`
- Analyzes employment rates by gender

**Function:** `occ_by_gender(dataframe)`
- Displays the top 10 occupation categories per gender

### 7. Racial & Ethnic Disparities
**Function:** `occupation_by_education(dataframe, is_include_unemployed=True)`
- Examines whether higher education levels correlate with specific occupations

**Function:** `edd_by_race(dataframe, parameter_object)`
- Compares education levels across different racial and ethnic groups

---
## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas datasets seaborn matplotlib
   ```
2. Run the script:
   ```bash
   python demographic.py
   ```

## Visualization Examples
- **Bar charts**: Name popularity, employment rates, and educational distributions.
- **Histograms**: Age distribution by education level.
- **Grouped analysis**: Comparisons of employment across demographics.

---
## Future Enhancements
- Add interactive visualizations using **Plotly**.
- Expand demographic filters.
- Improve performance on large datasets.

Feel free to contribute! 🚀

