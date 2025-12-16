import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os
from sklearn.ensemble import RandomForestRegressor
from shapely.geometry import box

# Configuration
TRAIN_DATA = 'data/ml_ready/train_2023.csv'
TEST_DATA = 'data/ml_ready/test_2024.csv'
FIGURES_DIR = 'reports/figures'

def analyze_errors():
    print("Loading data...")
    if not os.path.exists(TRAIN_DATA) or not os.path.exists(TEST_DATA):
        print("Data missing. Run ml_prep.py first.")
        return

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    # 1. Preprocessing (Same as before)
    time_mapping = {label: idx for idx, label in enumerate(train_df['Time_Period'].unique())}
    train_df['Time_Code'] = train_df['Time_Period'].map(time_mapping)
    test_df['Time_Code'] = test_df['Time_Period'].map(time_mapping)
    
    # 2. Train the Model
    print("Training model on 2023 data...")
    features = ['Latitude', 'Longitude', 'Day_Num', 'Time_Code']
    target = 'Incident_Count'
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df[target])
    
    # 3. Predict on 2024 Data
    print("Generating predictions for 2024...")
    test_df['Predicted'] = model.predict(test_df[features])
    
    # 4. Calculate Residuals (The Error)
    # Error = Actual - Predicted
    # Positive Error = Unexpected Crime Wave (Model Under-predicted)
    # Negative Error = Quieter than expected (Model Over-predicted)
    test_df['Error'] = test_df[target] - test_df['Predicted']
    
    # 5. Aggregate errors by Location
    # We want to see which ZONES are hardest to predict, regardless of time
    location_errors = test_df.groupby(['Latitude', 'Longitude'])['Error'].mean().reset_index()
    
    print("\nTop 5 Under-Predicted Zones (Unexpected Crime):")
    print(location_errors.sort_values('Error', ascending=False).head(5))
    
    # 6. Plot the Error Map
    plot_error_map(location_errors)

def plot_error_map(df):
    """Maps where the model was wrong."""
    print("Plotting Error Map...")
    
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    f, ax = plt.subplots(figsize=(12, 10))
    
    # We use a 'diverging' colormap (coolwarm)
    # Red = Under-predicted (Bad for safety)
    # Blue = Over-predicted (Wasted resources)
    # White = Perfect prediction
    gdf.plot(
        column='Error', 
        ax=ax, 
        alpha=0.7, 
        cmap='coolwarm', 
        markersize=50,
        legend=True,
        legend_kwds={'label': "Prediction Error (Actual - Predicted)"},
        vmin=-5, vmax=5 # Cap the colors so outliers don't wash out the map
    )
    
    # Add Basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except:
        pass

    # Force Zoom to Vegas (The fix we used before)
    min_lon, min_lat = -115.35, 35.95
    max_lon, max_lat = -114.95, 36.30
    bbox = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857)
    min_x, min_y, max_x, max_y = bbox.total_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.title('Model Residuals: Where did we get it wrong?', fontsize=15)
    ax.set_axis_off()
    
    output_path = f'{FIGURES_DIR}/10_error_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Error Analysis saved to: {output_path}")

if __name__ == "__main__":
    analyze_errors()