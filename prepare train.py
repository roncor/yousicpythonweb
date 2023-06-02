import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

def read_dataset(path, file):
    dataset = pd.read_csv(f"{path}/{file}")
    return dataset

def analyze_dataset(dataset):
    data_types = dataset.dtypes
    data_types_grouped = data_types.groupby(data_types).groups

    object_factors_count = []

    print("Attributes grouped by data type\n")
    for data_type, columns in data_types_grouped.items():
        print(f"Data Type: {data_type}")
        data_info = []
        for column in columns:
            if data_type == object:
                factors_count = dataset[column].nunique()
                data_info.append((column, dataset[column].dtypes, factors_count))
                if factors_count <= 100:
                    object_factors_count.append((column, factors_count))
            else:
                data_info.append((column, dataset[column].dtypes))

        print(tabulate(data_info, headers=['Attribute', 'Data Type', 'Factors' if data_type == object else ''], tablefmt='grid'))
        print()

        if data_type in [int, float]:
            null_values = dataset[columns].isnull().sum()
            if null_values.sum() > 0:
                print("Null values in the column(s):")
                print(null_values)
                missing_percentage = (null_values / len(dataset)) * 100
                print("Percentage of missing values:")
                print(missing_percentage)
            else:
                print("No null values in the column(s).")
            print()

    data_type_count = dataset.dtypes.value_counts().reset_index()
    data_type_count.columns = ['Data Type', 'Count']

    plt.figure(figsize=(10, 6))
    plt.pie(data_type_count['Count'], labels=data_type_count['Data Type'].astype(str), autopct='%1.1f%%')
    plt.title('Distribution of Data Types')
    plt.axis('equal')
    plt.savefig(f'{path}data_types_distribution.png')

    df_object_factors_count = pd.DataFrame(object_factors_count, columns=['Column', 'Factors Count'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Factors Count', y='Column', data=df_object_factors_count, errorbar=None)
    plt.title('Factors Count per Object Type Column')
    plt.xlabel('Factors Count')
    plt.ylabel('Column')
    plt.savefig(f'{path}factors_count_per_object_column.png')

    return data_type_count

def preprocess_dataset(dataset, path=''):
    new_dataset = dataset.copy()

    columns_to_fill = ['Name', 'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary', 'Credit_Mix', 'Payment_Behaviour', "Num_of_Loan", 
                       "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Delayed_Payment", "Monthly_Balance"]

    for column in columns_to_fill:
        print(f"Processing column: {column}")
        customer_id_to_mapping = new_dataset.groupby('Customer_ID')[column].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
        new_dataset.loc[new_dataset[column].isna(), column] = new_dataset.loc[new_dataset[column].isna(), 'Customer_ID'].map(customer_id_to_mapping)
        for customer_id, value in customer_id_to_mapping.items():
            new_dataset.loc[new_dataset['Customer_ID'] == customer_id, column] = value

    new_dataset.to_csv(f"{path}new_train.csv", index=False)

def print_column_data_types(dataset):
    column_data_types = dataset.dtypes
    for column, data_type in column_data_types.items():
        print(f"Column: {column}, Data Type: {data_type}")

def create_boxplots(dataset, path):
    numerical_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if 'Credit_Score' in numerical_columns:
        numerical_columns.remove('Credit_Score')
    if dataset['Credit_Score'].dtypes != object:
        print("'Credit_Score' is not of type 'object'. Converting it now.")
        dataset['Credit_Score'] = dataset['Credit_Score'].astype(str)

    for column in numerical_columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=dataset['Credit_Score'], y=dataset[column])
        plt.title(f'Boxplot of {column} for Each Credit_Score Category')
        plt.ylabel(column)
        plt.xlabel('Credit_Score')
        plt.savefig(f'{path}boxplot_{column}_credit_score.png')

def plot_attribute_distribution(dataset, attribute):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=dataset, x=attribute)
    plt.title(f'Distribution of {attribute}')
    plt.xlabel(attribute)
    plt.savefig(f'{path}distribution_{attribute}.png')

def remove_invalid_rows(dataset, columns):
    missing_columns = [col for col in columns if col not in dataset.columns]
    if missing_columns:
        print(f"Error: Column(s) {missing_columns} not found in the dataset.")
        return dataset

    cleaned_dataset = dataset.copy()

    for column in columns:
        cleaned_dataset[column] = cleaned_dataset[column].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
        cleaned_dataset[column] = pd.to_numeric(cleaned_dataset[column], errors="coerce")

        if column == "Num_Credit_Inquiries":
            cleaned_dataset = cleaned_dataset[cleaned_dataset[column] <= 100]
        elif column == "Age":
            cleaned_dataset = cleaned_dataset[cleaned_dataset[column] <= 100]  # Remove values higher than 100 for 'Age'
        else:
            cleaned_dataset = cleaned_dataset[cleaned_dataset[column] >= 0]

    return cleaned_dataset

# Read the dataset
path = "replace the path"
file = "train.csv"
dataset = read_dataset(path, file)

# Analyze dataset
result = analyze_dataset(dataset)
print(result)

# Preprocess data
#preprocess_dataset(dataset, path)

# Get new data
file = "new_train.csv"
dataset_new = read_dataset(path, file)

# Create boxplots
#create_boxplots(dataset_new, path)

# Plot distribution of attributes
attributes_to_plot = ['Age', 'Annual_Income', 'Changed_Credit_Limit', 'Amount_invested_monthly', 'Outstanding_Debt', 'Delay_from_due_date', 'Num_of_Loan', 'Num_Bank_Accounts', 'Num_Credit_Inquiries', 'Monthly_Balance']
for attribute in attributes_to_plot:
    plot_attribute_distribution(dataset_new, attribute)

# Remove invalid rows
cleaned_df = remove_invalid_rows(dataset_new, ["Age", "Annual_Income", "Changed_Credit_Limit", "Amount_invested_monthly", "Outstanding_Debt", "Delay_from_due_date", "Num_of_Loan", "Num_Bank_Accounts", "Num_Credit_Inquiries", "Monthly_Balance"])

# Print column data types
print_column_data_types(cleaned_df)

# Save the cleaned dataframe
cleaned_df.to_csv(f"{path}cleaned_df.csv", index=False)
