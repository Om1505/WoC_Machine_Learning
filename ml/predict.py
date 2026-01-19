import joblib
import numpy as np

def test_predictions():
    model = joblib.load('ml/models/placement_model.pkl')
    
    # Define two test cases (Ensure the number of features matches your X_train)
    # [IQ, CGPA, Internship, Academic_Score, ..., Academic_Power]
    
    perfect_student = [120, 9.8, 1, 10, 10, 9, 9, 10, 1372] # High CGPA/IQ/Intern
    struggling_student = [80, 5.5, 0, 4, 3, 2, 5, 1, 440]   # Low CGPA/IQ/No Intern

    for student in [perfect_student, struggling_student]:
        data = np.array(student).reshape(1, -1)
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]
        
        status = "Placed" if pred == 1 else "Not Placed"
        print(f"Student Data: {student}")
        print(f"Result: {status} | Confidence: {prob*100:.2f}%\n")

if __name__ == "__main__":
    test_predictions()