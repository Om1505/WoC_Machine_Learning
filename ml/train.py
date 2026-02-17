import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from preprocess import preprocess_for_xgboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def objective_classification(trial, X_train, X_test, y_train, y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test))

def objective_regression(trial, X_train, X_test, y_train, y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'objective': 'reg:squarederror'
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    return mean_absolute_error(y_test, model.predict(X_test))

def train_dual_models():
    X_train, X_test, y_p_train, y_p_test, y_s_train, y_s_test = preprocess_for_xgboost()

    print("--- Optimizing Placement Classifier ---")
    study_clf = optuna.create_study(direction='maximize')
    study_clf.optimize(lambda t: objective_classification(t, X_train, X_test, y_p_train, y_p_test), n_trials=20)
    
    print("--- Optimizing Salary Regressor ---")
    study_reg = optuna.create_study(direction='minimize')
    study_reg.optimize(lambda t: objective_regression(t, X_train, X_test, y_s_train, y_s_test), n_trials=20)

    clf = xgb.XGBClassifier(**study_clf.best_params, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_p_train)
    
    reg = xgb.XGBRegressor(**study_reg.best_params)
    reg.fit(X_train, y_s_train)

    y_p_pred = clf.predict(X_test)
    y_s_pred = reg.predict(X_test)
    
    acc = accuracy_score(y_p_test, y_p_pred)
    mae = mean_absolute_error(y_s_test, y_s_pred)
    r2 = r2_score(y_s_test, y_s_pred)

    print(f"\nPlacement Accuracy: {acc:.4f} | Salary MAE: ₹{mae:.2f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    xgb.plot_importance(clf, ax=ax1, color='#710193', title='Placement Importance')
    xgb.plot_importance(reg, ax=ax2, color='#FF7518', title='Salary Importance')
    plt.savefig(os.path.join(OUTPUT_DIR, 'salary_importance.png'))
    plt.close()

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'placement_model.pkl'))
    joblib.dump(reg, os.path.join(MODEL_DIR, 'salary_model.pkl'))
    
    with open(os.path.join(OUTPUT_DIR, 'training_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Placement Accuracy: {acc:.4f}\nSalary MAE: ₹{mae:.2f}\nSalary R2: {r2:.4f}")

if __name__ == "__main__":
    train_dual_models()