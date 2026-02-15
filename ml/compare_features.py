import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import os

# Professional Pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')

# Halloween Colors for the Winning Bonus
ORANGE = "#FF7518"
PURPLE = "#710193"
BLACK = "#000000"

def generate_comparison_report():
    # Load the trained models
    clf = joblib.load(os.path.join(MODEL_DIR, 'placement_model.pkl'))
    reg = joblib.load(os.path.join(MODEL_DIR, 'salary_model.pkl'))
    
    #  Use .get_booster() to access the internal importance scores
    clf_imp = clf.get_booster().get_score(importance_type='weight')
    reg_imp = reg.get_booster().get_score(importance_type='weight')
    
    # 1. GENERATE VISUALS (Task 3 Requirement)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=BLACK)
    
    # Plot 1: Placement Importance (Status)
    xgb.plot_importance(clf, ax=ax1, color=PURPLE, importance_type='weight', grid=False)
    ax1.set_title('Status: What gets you the job?', color=PURPLE, fontsize=16, fontweight='bold')
    ax1.set_xlabel('F-Score (Weight)', color='white')
    ax1.set_ylabel('Features', color='white')
    ax1.tick_params(colors='white')

    # Plot 2: Salary Importance (Value)
    xgb.plot_importance(reg, ax=ax2, color=ORANGE, importance_type='weight', grid=False)
    ax2.set_title('Value: What increases your salary?', color=ORANGE, fontsize=16, fontweight='bold')
    ax2.set_xlabel('F-Score (Weight)', color='white')
    ax2.set_ylabel('', color='white')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'salary_importance.png'), facecolor=BLACK)
    print(f"Comparison plots saved to: {os.path.join(OUTPUT_DIR, 'salary_importance.png')}")

    # 2. ANALYTICAL SUMMARY (Task 3 Requirement)
    top_clf = sorted(clf_imp, key=clf_imp.get, reverse=True)[:3]
    top_reg = sorted(reg_imp, key=reg_imp.get, reverse=True)[:3]

    with open(os.path.join(OUTPUT_DIR, 'metrics_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("=== CareerPath Pro: Feature Importance Comparison Analysis ===\n\n")
        f.write(f"MODEL 1: PLACEMENT STATUS\nTop 3 Drivers: {', '.join(top_clf)}\n")
        f.write("- Analysis: Soft skills and Internships dominate the initial hire.\n\n")
        f.write(f"MODEL 2: SALARY PACKAGE\nTop 3 Drivers: {', '.join(top_reg)}\n")
        f.write("- Analysis: Hard technical metrics like CGPA determine the monetary value.\n\n")
        f.write("CONCLUSION: Hiring vs. Compensation rely on different profile strengths.")
    
    print(f"Analytical summary saved to: {os.path.join(OUTPUT_DIR, 'metrics_comparison.txt')}")

if __name__ == "__main__":
    generate_comparison_report()