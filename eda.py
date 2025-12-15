import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_PATH = 'data/cleaned_lvmpd_incidents.parquet'
FIGURES_DIR = 'reports/figures'

def setup_plotting():
    """Sets specific style preferences for charts."""
    sns.set_theme(style="whitegrid")
    # Create directory for saving images if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
# End of setup_plotting 

def load_data(filepath):
    """Loads the cleaned parquet data."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please run data_cleaning.py first.")
        return None
    return pd.read_parquet(filepath)
# End of load_data

def plot_crime_distribution(df):
    """Plot 1: Bar chart of incident counts by Crime Category."""
    plt.figure(figsize=(12, 6))
    
    # Order categories by frequency
    order = df['Crime_Category'].value_counts().index
    # Create count plot
    sns.countplot(y='Crime_Category', data=df, order=order, palette='viridis')
    plt.title('Distribution of Incident Types', fontsize=15)
    plt.xlabel('Number of Calls')
    plt.ylabel('Category')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/01_crime_category_distribution.png')
    print("Saved: 01_crime_category_distribution.png")
    plt.close()
# End of plot_crime_distribution

def plot_temporal_patterns(df):
    """Plot 2: Incidents by Time Period and Weekend Status."""
    plt.figure(figsize=(10, 6))
    # Define order for time periods
    time_order = ['Morning', 'Afternoon', 'Evening', 'Late_Night']
    # Create count plot with hue for weekend status
    sns.countplot(x='Time_Period', data=df, order=time_order, hue='Is_Weekend', palette='coolwarm')
    plt.title('Incidents by Time of Day (Weekend vs Weekday)', fontsize=15)
    plt.xlabel('Time Period')
    plt.ylabel('Number of Calls')
    plt.legend(title='Is Weekend?', labels=['Weekday (Mon-Fri)', 'Weekend (Sat-Sun)'])
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/02_temporal_patterns.png')
    print("Saved: 02_temporal_patterns.png")
    plt.close()

def plot_spatial_sanity_check(df):
    """Plot 3: Scatter map of coordinates to identify outliers."""
    plt.figure(figsize=(10, 10))
    
    # Scatter plot with low alpha for density visualization
    sns.scatterplot(
        x='Longitude', 
        y='Latitude', 
        data=df, 
        alpha=0.1, 
        s=1, 
        hue='Crime_Category', 
        legend=False # Legend is too big for a scatter map usually
    )
    
    plt.title('Spatial Distribution of Incidents (Sanity Check)', fontsize=15)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # If points are way off the expected area, set limits to zoom in
    # plt.xlim(-115.4, -114.8) 
    # plt.ylim(35.9, 36.4)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/03_spatial_sanity_check.png')
    print("Saved: 03_spatial_sanity_check.png")
    plt.close()

# Plot 4: Normalized Temporal Analysis
def plot_normalized_temporal(df):
    """
    Plots the Average Incidents per Day (normalizing for 5 weekdays vs 2 weekend days).
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Calculate the raw counts
    counts = df.groupby(['Time_Period', 'Is_Weekend'], observed=False).size().reset_index(name='Total_Counts')
    
    # 2. Normalize: Divide Weekday totals by 5, Weekend totals by 2
    # Logic: If Is_Weekend == 1 (True), divide by 2. Else divide by 5.
    counts['Daily_Average'] = counts.apply(
        lambda row: row['Total_Counts'] / 2 if row['Is_Weekend'] == 1 else row['Total_Counts'] / 5, 
        axis=1
    )
    
    # 3. Plot the Averages
    time_order = ['Morning', 'Afternoon', 'Evening', 'Late_Night']
    
    sns.barplot(
        x='Time_Period', 
        y='Daily_Average', 
        hue='Is_Weekend', 
        data=counts, 
        order=time_order, 
        palette='coolwarm'
    )
    
    plt.title('Average Daily Incidents (Normalized)', fontsize=15)
    plt.ylabel('Average Calls Per Day')
    plt.xlabel('Time Period')
    plt.legend(title='Is Weekend?', labels=['Weekday (Avg of 5 days)', 'Weekend (Avg of 2 days)'])
    
    plt.tight_layout()
    plt.savefig('reports/figures/04_normalized_temporal.png')
    print("Saved: 04_normalized_temporal.png")
    plt.close()

if __name__ == "__main__":
    setup_plotting()
    
    print("Loading data...")
    df = load_data(INPUT_PATH)
    
    if df is not None:
        print(f"Data Loaded. Rows: {len(df)}")
        
        print("Generating Plot 1: Crime Distribution...")
        plot_crime_distribution(df)
        
        print("Generating Plot 2: Temporal Patterns...")
        plot_temporal_patterns(df)
        
        print("Generating Plot 3: Spatial Map...")
        plot_spatial_sanity_check(df)

        print("Generating Plot 4: Normalized Temporal Patterns...")
        plot_normalized_temporal(df)
        
        print(f"\nEDA Complete. Check the '{FIGURES_DIR}' folder for images.")