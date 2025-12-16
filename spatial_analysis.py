import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_PATH = 'data/cleaned_lvmpd_incidents.parquet'
FIGURES_DIR = 'reports/figures'

def load_data(filepath):
    """Loads the cleaned parquet data."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    return pd.read_parquet(filepath)

def plot_density_map(df):
    """
    Creates a 2D KDE (Kernel Density Estimation) plot.
    This shows the 'intensity' of incidents across the city.
    """
    print("Preparing data for Density Map...")
    
    # 1. Filter Outliers (Crucial for KDE)
    # We roughly filter for Las Vegas coordinates to stop the plot from squishing
    # if there is a random point at (0,0).
    vegas_df = df[
        (df['Latitude'] > 35.9) & (df['Latitude'] < 36.4) & 
        (df['Longitude'] > -115.4) & (df['Longitude'] < -114.8)
    ].copy()
    
    print(f"Filtered to {len(vegas_df)} incidents inside Vegas bounds.")

    # 2. Downsample for Speed
    # KDE is very slow on 100k+ points. We sample 30k points to get the 'shape' of the data.
    if len(vegas_df) > 30000:
        print("Downsampling to 30,000 points for faster plotting...")
        plot_data = vegas_df.sample(30000, random_state=42)
    else:
        plot_data = vegas_df

    print("Generating Density Plot (this may take a moment)...")
    
    # 3. Create the Plot
    f, ax = plt.subplots(figsize=(12, 10))
    
    # A. Draw the 'City Shape' using a light scatter plot
    sns.scatterplot(
        x='Longitude', y='Latitude', 
        data=plot_data, 
        s=1, color='grey', alpha=0.1, ax=ax
    )
    
    # B. Draw the 'Hot Spots' using KDE (The Heatmap)
    # levels=20 gives us 20 'steps' of intensity
    # fill=True fills the contours with color
    sns.kdeplot(
        x='Longitude', y='Latitude', 
        data=plot_data, 
        cmap='inferno', 
        fill=True, 
        alpha=0.6, 
        levels=20, 
        thresh=0.05, # Hides the lowest 5% density (cleans up the background)
        ax=ax
    )

    ax.set_title('Las Vegas Crime Density (Seaborn KDE)', fontsize=20)
    ax.set_axis_off() # Hide the box, just show the map shape
    
    output_path = f'{FIGURES_DIR}/05_seaborn_density.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_comparative_hotspots(df):
    """
    Uses Seaborn's FacetGrid to compare hotspots for different crime types side-by-side.
    """
    print("Generating Comparative Hot Spots (Violent vs Property)...")
    
    # Filter for the two main categories we care about
    target_crimes = ['Violent_Crime', 'Property_Crime']
    subset = df[df['Crime_Category'].isin(target_crimes)].copy()
    
    # Filter bounds
    subset = subset[
        (subset['Latitude'] > 35.9) & (subset['Latitude'] < 36.4) & 
        (subset['Longitude'] > -115.4) & (subset['Longitude'] < -114.8)
    ]
    
    # Downsample if needed
    if len(subset) > 20000:
        subset = subset.sample(20000, random_state=42)

    # Create a FacetGrid (One plot per category)
    g = sns.FacetGrid(subset, col="Crime_Category", height=8, sharex=True, sharey=True)
    
    # Map the KDE plot onto the grid
    g.map_dataframe(
        sns.kdeplot, 
        x='Longitude', 
        y='Latitude', 
        fill=True, 
        cmap='rocket', # 'rocket' is a good palette for intensity
        alpha=0.7,
        levels=15
    )
    
    # Add titles
    g.set_titles(col_template="{col_name} Hotspots", size=15)
    g.axes.flatten()[0].set_ylabel('Latitude')
    g.axes.flatten()[0].set_xlabel('Longitude')
    g.axes.flatten()[1].set_xlabel('Longitude')

    output_path = f'{FIGURES_DIR}/06_comparative_hotspots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Create output dir if needed
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load data
    df = load_data(INPUT_PATH)
    
    if df is not None:
        # Plot 1: Overall Density
        plot_density_map(df)
        
        # Plot 2: Compare Categories
        plot_comparative_hotspots(df)