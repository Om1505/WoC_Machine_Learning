import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'college_student_placement_dataset.csv')

def preprocess_for_xgboost(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    
    df['Academic_Power'] = df['CGPA'] * df['IQ']
    
    # We will encode ALL object columns to avoid the 'College_ID' error
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    for col in object_cols:
        # We drop unique identifiers like Student_ID because they have no predictive power
        if 'ID' in col or 'id' in col:
            df.drop(columns=[col], inplace=True)
            continue
            
        # For meaningful categories (like Internship_Experience), we encode them
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(model_dir, f'label_encoder_{col}.pkl'))

    # Prepare features and target
    X = df.drop(columns=['Placement', 'Salary'], errors='ignore')
    y = df['Placement']

    # Task 4: Stratified split 
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if __name__ == "__main__":
    preprocess_for_xgboost()