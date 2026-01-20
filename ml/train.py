import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os
from preprocess import preprocess_for_xgboost

# Professional Pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def objective(trial, X_train, X_test, y_train, y_test):
    # Defining parameter ranges based on Hyperparameter Guide 
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    # We optimize for F1-Score to balance Precision and Recall 
    return f1_score(y_test, preds)

def train_model():
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_xgboost()
    print(X_train.columns)

    # 1. Optuna Hyperparameter Tuning
    print("--- Starting Optuna Tuning ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=50)

    print(f"Best trial F1-Score: {study.best_value:.4f}")
    print("Best parameters:", study.best_params)

    # 2. Train final model with best parameters
    best_model = xgb.XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train, y_train)

    # 3. Evaluation 
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- Final Training Metrics ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

    # 4. Save results [cite: 152]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'training_metrics.txt'), 'w') as f:
        f.write(f"Best Params: {study.best_params}\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}")

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(best_model, color='#FF7518', importance_type='weight') 
    plt.title('Optimized XGBoost Importance (Halloween Edition)', color='#710193')
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'placement_model.pkl'))
    print(f"Optimized model saved successfully in: {MODEL_DIR}")

if __name__ == "__main__":
    train_model()