import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import contextily as ctx
import os
from sklearn.ensemble import RandomForestRegressor

# Configuration
TRAIN_DATA = 'data/ml_ready/train_2023.csv'
TEST_DATA = 'data/ml_ready/test_2024.csv'
FIGURES_DIR = 'reports/figures'

def load_and_train():
    """
    Retrains the model on ALL available data (2023) to be ready for user queries.
    In a real app, we would save/load the model using joblib, but retraining 
    on the fly is fine for this demo.
    """
    print("Initializing Recommendation Engine...")
    df = pd.read_csv(TRAIN_DATA)
    
    # Preprocessing
    time_mapping = {label: idx for idx, label in enumerate(df['Time_Period'].unique())}
    df['Time_Code'] = df['Time_Period'].map(time_mapping)
    
    # Train
    features = ['Latitude', 'Longitude', 'Day_Num', 'Time_Code']
    target = 'Incident_Count'
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df[features], df[target])
    
    return model, time_mapping, df

def get_recommendations(model, time_mapping, df_ref, day_name, time_period):
    """
    Generates a 'Patrol Map' for a specific day and time.
    """
    print(f"\n--- Generating Patrol Plan for {day_name} {time_period} ---")
    
    # Convert inputs to model format
    days = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    day_num = days.get(day_name)
    time_code = time_mapping.get(time_period)
    
    if day_num is None or time_code is None:
        print("Error: Invalid Day or Time Period.")
        return

    # 1. Get all unique locations (Hexagons) from our reference data
    # We want to predict the risk for EVERY zone in the city for this specific time
    unique_locations = df_ref[['h3_polyfill', 'Latitude', 'Longitude']].drop_duplicates()
    
    # 2. Create a 'Hypothetical' dataset for this specific time slot
    unique_locations['Day_Num'] = day_num
    unique_locations['Time_Code'] = time_code
    
    # 3. Predict Risk
    features = ['Latitude', 'Longitude', 'Day_Num', 'Time_Code']
    unique_locations['Predicted_Risk'] = model.predict(unique_locations[features])
    
    # 4. Rank the Zones (Highest Risk First)
    top_hotspots = unique_locations.sort_values('Predicted_Risk', ascending=False).head(10)
    
    print("\nTop 5 Recommended Patrol Zones:")
    print(top_hotspots[['h3_polyfill', 'Predicted_Risk']].head(5))
    
    return unique_locations, top_hotspots

def plot_recommendation(all_zones, top_zones, day_name, time_period):
    """Plots the recommendation on a map."""
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        all_zones, 
        geometry=gpd.points_from_xy(all_zones.Longitude, all_zones.Latitude),
        crs="EPSG:4326"
    )
    
    # Reproject for map (Web Mercator)
    gdf_web = gdf.to_crs(epsg=3857)
    
    f, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all zones (color by risk)
    gdf_web.plot(
        column='Predicted_Risk', 
        ax=ax, 
        alpha=0.6, 
        cmap='Reds', 
        markersize=50,
        legend=True,
        legend_kwds={'label': "Predicted Incident Volume"}
    )
    
    # Circle the Top 10 Targets
    top_gdf = gpd.GeoDataFrame(
        top_zones, 
        geometry=gpd.points_from_xy(top_zones.Longitude, top_zones.Latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    top_gdf.plot(ax=ax, color='none', edgecolor='blue', linewidth=2, markersize=200, label='Top Priority')
    
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except:
        pass
        
    # --- NEW: FORCE ZOOM TO LAS VEGAS VALLEY ---
    # We create a bounding box in Lat/Lon and convert it to Web Mercator
    # Approx Bounds: West: -115.35, East: -114.95, South: 35.95, North: 36.30
    min_lon, min_lat = -115.35, 35.95
    max_lon, max_lat = -114.95, 36.30
    
    # Convert these corners to the map's projection (EPSG:3857)
    from shapely.geometry import box
    bbox = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857)
    
    # Set the plot limits to this box
    min_x, min_y, max_x, max_y = bbox.total_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # -------------------------------------------

    plt.title(f'Recommended Patrol Deployment: {day_name} {time_period}', fontsize=15)
    ax.set_axis_off()
    
    output_path = f'{FIGURES_DIR}/09_deployment_plan_{day_name}_{time_period}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {output_path}")

if __name__ == "__main__":
    # Load
    model, mapping, df_ref = load_and_train()
    
    # --- USER SCENARIO: SATURDAY LATE NIGHT ---
    # You can change these variables to test different scenarios
    TARGET_DAY = 'Saturday'
    TARGET_TIME = 'Late_Night'
    
    all_risks, top_risks = get_recommendations(model, mapping, df_ref, TARGET_DAY, TARGET_TIME)
    
    plot_recommendation(all_risks, top_risks, TARGET_DAY, TARGET_TIME)