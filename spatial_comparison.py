import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
FILE_2023 = 'data/cleaned_lvmpd_2023.parquet'
FILE_2024 = 'data/cleaned_lvmpd_2024.parquet'
FIGURES_DIR = 'reports/figures'

def load_and_prep(filepath, year_label):
    """Loads data and filters to Vegas bounds."""
    if not os.path.exists(filepath):
        print(f"Missing file: {filepath}")
        return None
    
    df = pd.read_parquet(filepath)
    
    # Filter strictly to Vegas Valley to ensure maps match perfectly
    # (Removes outliers that would skew the zoom level)
    df = df[
        (df['Latitude'] > 35.9) & (df['Latitude'] < 36.4) & 
        (df['Longitude'] > -115.35) & (df['Longitude'] < -114.9)
    ].copy()
    
    # Add a column for plotting labels
    df['Year'] = year_label
    
    # Downsample for speed if needed (Seaborn is slow with 100k+ points)
    if len(df) > 30000:
        return df.sample(30000, random_state=42)
    return df

def plot_year_comparison(df_combined):
    """Plots 2023 vs 2024 density side-by-side."""
    print("Generating Year-over-Year Comparison...")
    
    g = sns.FacetGrid(df_combined, col="Year", height=8)
    
    # Create the density heatmaps
    g.map_dataframe(
        sns.kdeplot, 
        x='Longitude', 
        y='Latitude', 
        fill=True, 
        cmap='inferno', # 'inferno' is great for heatmaps
        levels=20, 
        thresh=0.05,
        alpha=0.7
    )
    
    # Add titles and labels
    g.set_titles(col_template="{col_name} Hotspots", size=20)
    g.axes.flatten()[0].set_ylabel('Latitude')
    
    # Save
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = f'{FIGURES_DIR}/07_year_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("Loading 2023 Data...")
    df_23 = load_and_prep(FILE_2023, '2023')
    
    print("Loading 2024 Data...")
    df_24 = load_and_prep(FILE_2024, '2024')
    
    if df_23 is not None and df_24 is not None:
        # Combine them into one dataframe for Seaborn plotting
        combined_df = pd.concat([df_23, df_24])
        
        plot_year_comparison(combined_df)