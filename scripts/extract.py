import pandas as pd
import numpy as np
import os

def clean_and_prepare_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle Missing Values 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Data Type Correction 
    binary_mapping = {'Yes': 1, 'No': 0}
    if 'Internship_Experience' in df.columns:
        df['Internship_Experience'] = df['Internship_Experience'].map(binary_mapping)
    if 'Placement' in df.columns:
        df['Placement'] = df['Placement'].map(binary_mapping)
    
    # IQR Outlier Capping 
    for col in ['CGPA', 'IQ']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.where(df[col] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR, df[col])
            df[col] = np.where(df[col] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, df[col])

    # Advanced Feature Engineering 
    # Create the 'Salary' feature before training 

    df['Academic_Efficiency'] = (df['CGPA'] * 10) + (df['IQ'] / 10)

    df['Salary'] = (df['CGPA'] * 40000) + (df['Internship_Experience'] * 100000) + (df['IQ'] * 500)
    df.loc[df['Placement'] == 0, 'Salary'] = 0 # 0 salary if not placed

    print(df.sample(5))  
    
    df.to_csv('data/cleaned_data.csv', index=False)
    return df

if __name__ == "__main__":
    clean_and_prepare_data('data/college_student_placement_dataset.csv')