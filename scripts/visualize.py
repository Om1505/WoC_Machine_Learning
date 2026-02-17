import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

HALLOWEEN_COLORS = ["#000000", "#710193", "#FF7518"] 
CUSTOM_CMAP = LinearSegmentedColormap.from_list("halloween", HALLOWEEN_COLORS)

def setup_directories():
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures')

def generate_visualizations(data_path):
    df = pd.read_csv(data_path)
    
    plt.figure(figsize=(10, 8))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap=CUSTOM_CMAP, fmt='.2f')
    plt.title('Feature Correlation Heatmap (Halloween Edition)')
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df['CGPA'], kde=True, color="#FF7518", ax=axes[0])
    axes[0].set_title('CGPA Distribution (Outlier-Capped)')
    
    sns.histplot(df['IQ'], kde=True, color="#710193", ax=axes[1])
    axes[1].set_title('IQ Score Distribution (Outlier-Capped)')
    
    plt.savefig('reports/figures/distribution_plots.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Placement', y='CGPA', data=df, palette=[HALLOWEEN_COLORS[1], HALLOWEEN_COLORS[2]])
    plt.title('Placement Status vs. Academic Performance')
    plt.savefig('reports/figures/placement_boxplot.png')
    plt.close()

    print("Successfully generated all reports in /reports/figures/")

if __name__ == "__main__":
    setup_directories()
    generate_visualizations('data/cleaned_data.csv')