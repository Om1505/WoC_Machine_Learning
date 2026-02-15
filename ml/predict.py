import joblib
import numpy as np
import os

# Set up professional pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')

def predict_student_outcome(student_features):
    # Load optimized models and encoders
    clf = joblib.load(os.path.join(MODEL_DIR, 'placement_model.pkl'))
    reg = joblib.load(os.path.join(MODEL_DIR, 'salary_model.pkl'))
    
    # Reshape input for prediction
    data = np.array(student_features).reshape(1, -1)
    
    # 1. Predict Placement Status (Classification)
    placement_pred = clf.predict(data)[0]
    placement_prob = clf.predict_proba(data)[0][1]
    
    # 2. Predict Salary (Regression)
    # Logic: If model predicts no placement, salary is 0
    if placement_pred == 1:
        salary_pred = reg.predict(data)[0]
    else:
        salary_pred = 0.0
        
    return placement_pred, placement_prob, salary_pred

def run_test_cases():
    # Feature Order must match X_train:
    # [IQ, Prev_Sem_Result, CGPA, Academic_Performance, Internship_Experience, 
    #  Extra_Curricular_Score, Communication_Skills, Projects_Completed, Academic_Power]
    
    test_cases = [
        {
            "name": "High Achiever (Spooky Success)",
            "data": [140, 9.5, 9.8, 10, 1, 10, 10, 10, 1372.0]
        },
        {
            "name": "Struggling Student (Grave Outlook)",
            "data": [85, 5.0, 5.8, 4, 0, 3, 4, 1, 493.0]
        }
    ]

    print("--- CareerPath Pro: Final Prediction Report ---")
    for case in test_cases:
        is_placed, prob, salary = predict_student_outcome(case['data'])
        
        status = "PLACED" if is_placed == 1 else "NOT PLACED"
        
        print(f"\nStudent: {case['name']}")
        print(f"Features: {case['data']}")
        print(f"Status: {status} (Confidence: {prob*100:.2f}%)")
        # Formatting salary as currency for professional output
        print(f"Estimated Salary: ₹{salary:,.2f}")
        print("-" * 45)

if __name__ == "__main__":
    run_test_cases()