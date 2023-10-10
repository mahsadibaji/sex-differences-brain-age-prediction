import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
dataset1 = pd.read_csv('/path/to/dataset1.csv')
dataset2 = pd.read_csv('/path/to/dataset2.csv')


def create_test_sets(data, n, test_size=0.2, random_state=42):
    """
    Split the data into male, female test sets and the rest.
    Stratification based on vandor and magnetic field strength information.
    """
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    male_train, male_test = train_test_split(male_data, test_size=n, stratify=male_data[['vendor', 'field']], random_state=random_state)
    female_train, female_test = train_test_split(female_data, test_size=n, stratify=female_data[['vendor', 'field']], random_state=random_state)

    return male_test, female_test, pd.concat([male_train, female_train])


def create_sex_dev_sets(data):
    """
    Split data by sex and return sex specific train and validation sets.
    Stratification based on vandor and magnetic field strength information.
    """
    males = data[data['sex'] == 'M']
    females = data[data['sex'] == 'F']
    
    male_train, male_val = train_test_split(males, test_size=0.2, stratify=males[['vendor', 'field']], random_state=42)
    female_train, female_val = train_test_split(females, test_size=0.2, stratify=females[['vendor', 'field']], random_state=42)
    
    return male_train, male_val, female_train, female_val


# Split the datasets into test sets and the rest
dataset1_male_test, dataset1_female_test, dataset1_rest = create_test_sets(dataset1, 30)
dataset2_male_test, dataset2_female_test, dataset2_rest = create_test_sets(dataset2, 30)

# Further split the rest of the data into train and validation sets
dataset1_male_train, dataset1_male_val, dataset1_female_train, dataset1_female_val = create_sex_dev_sets(dataset1_rest)
dataset2_male_train, dataset2_male_val, dataset2_female_train, dataset2_female_val = create_sex_dev_sets(dataset2_rest)

# Concatenate rest of the data from both datasets
combined_data = pd.concat([dataset1_rest, dataset2_rest])

# Create train and validation sets for the combined data
combined_train, combined_val = train_test_split(combined_data, test_size=0.2, stratify=combined_data[['sex', 'vendor', 'field']], random_state=42)

# Further split the combined data into sex specific train and validation sets
combined_male_train, combined_male_val, combined_female_train, combined_female_val = create_sex_dev_sets(combined_data)

# Saving the datasets to csv
dataset1_male_test.to_csv('dataset1_male_test.csv', index=False)
dataset1_female_test.to_csv('dataset1_female_test.csv', index=False)
dataset2_male_test.to_csv('dataset2_male_test.csv', index=False)
dataset2_female_test.to_csv('dataset2_female_test.csv', index=False)

dataset1_male_train.to_csv('dataset1_male_train.csv', index=False)
dataset1_male_val.to_csv('dataset1_male_val.csv', index=False)
dataset1_female_train.to_csv('dataset1_female_train.csv', index=False)
dataset1_female_val.to_csv('dataset1_female_val.csv', index=False)

dataset2_male_train.to_csv('dataset2_male_train.csv', index=False)
dataset2_male_val.to_csv('dataset2_male_val.csv', index=False)
dataset2_female_train.to_csv('dataset2_female_train.csv', index=False)
dataset2_female_val.to_csv('dataset2_female_val.csv', index=False)

combined_train.to_csv('combined_train.csv', index=False)
combined_val.to_csv('combined_val.csv', index=False)
combined_male_train.to_csv('combined_male_train.csv', index=False)
combined_male_val.to_csv('combined_male_val.csv', index=False)
combined_female_train.to_csv('combined_female_train.csv', index=False)
combined_female_val.to_csv('combined_female_val.csv', index=False)
