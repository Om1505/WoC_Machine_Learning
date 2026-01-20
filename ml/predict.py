import joblib
import numpy as np

def test_predictions():
    # Load the trained model from the required directory
    model = joblib.load('ml/models/placement_model.pkl')

    # Define two test cases matching the X_train column order
    perfect_student = [140, 9.5, 9.8, 10, 1, 10, 10, 10, 1372.0] # High Stats + Intern
    struggling_student = [80, 5.0, 5.5, 4, 0, 3, 4, 1, 440.0]    # Low Stats + No Intern

    test_cases = [
        ("Perfect Student", perfect_student),
        ("Struggling Student", struggling_student)
    ]

    for name, student in test_cases:
        # Reshape for a single prediction as required for the task 
        data = np.array(student).reshape(1, -1)
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]
        
        # Result logic based on classification 
        status = "Placed" if pred == 1 else "Not Placed"
        print(f"--- {name} ---")
        print(f"Student Data: {student}")
        print(f"Result: {status} | Confidence: {prob*100:.2f}%\n")

if __name__ == "__main__":
    test_predictions()