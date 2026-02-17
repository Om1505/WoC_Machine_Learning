import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'college_student_placement_dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def synthesize_salary(df):
    np.random.seed(42)
    
    df['Salary'] = (df['CGPA'] * 50000) + (df['IQ'] * 1000) + (df['Internship_Experience'] * 150000)
    
    noise = np.random.normal(0, 25000, df.shape[0])
    df['Salary'] = df['Salary'] + noise
    
    df.loc[df['Placement'] == 0, 'Salary'] = 0
    return df

def preprocess_for_xgboost(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    le = LabelEncoder()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if 'Internship_Experience' in df.columns:
        df['Internship_Experience'] = le.fit_transform(df['Internship_Experience'])
    
    if 'Placement' in df.columns:
        df['Placement'] = le.fit_transform(df['Placement'])
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    df = synthesize_salary(df)
    
    df['Academic_Power'] = df['CGPA'] * df['IQ']

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if 'ID' in col or 'id' in col:
            df.drop(columns=[col], inplace=True)

    X = df.drop(columns=['Placement', 'Salary'], errors='ignore')
    y_placement = df['Placement']
    y_salary = df['Salary']

    return train_test_split(X, y_placement, y_salary, test_size=0.2, stratify=y_placement, random_state=42)

if __name__ == "__main__":
    X_train, X_test, yp_train, yp_test, ys_train, ys_test = preprocess_for_xgboost()
    print("Success! Multi-target data split ready.")
    print(f"Features used: {list(X_train.columns)}")