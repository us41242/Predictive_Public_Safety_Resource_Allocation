import pandas as pd
import geopandas as gpd
import h3pandas
import os

# Configuration
# Resolution 8 = roughly neighborhood size (good for prediction)
H3_RESOLUTION = 8 
INPUT_2023 = 'data/cleaned_lvmpd_2023.parquet'
INPUT_2024 = 'data/cleaned_lvmpd_2024.parquet'
OUTPUT_DIR = 'data/ml_ready'

def process_year(filepath, year_label):
    """
    Turns raw points into a 'Risk Table':
    Row = (Location, Day of Week, Time of Day) -> Target = Incident Count
    """
    print(f"Processing {year_label} data...")
    
    # 1. Load Data
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (Not Found)")
        print("Make sure you are running this from the project root folder!")
        return None
        
    try:
        # Load the data
        gdf = gpd.read_parquet(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # 2. Assign Hexagon IDs (H3)
    # Check/Set CRS
    if gdf.crs is None:
        print("  Warning: CRS missing. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)
    
    print("  Assigning spatial grid...")
    # This puts the H3 ID in the INDEX
    try:
        gdf_hex = gdf.h3.geo_to_h3(H3_RESOLUTION)
    except Exception as e:
        print(f"Error during hex grid generation: {e}")
        return None
    
    # Move H3 ID from Index to Column
    gdf_hex = gdf_hex.reset_index()
    
    # Find the column that contains the H3 index (it usually starts with 'h3_')
    h3_cols = [c for c in gdf_hex.columns if c.startswith('h3_')]
    
    if not h3_cols:
        print("  Error: Could not find H3 index column after processing.")
        return None
        
    h3_col_name = h3_cols[0]
    print(f"  Renaming column '{h3_col_name}' to 'h3_polyfill'")
    
    # Rename it to 'h3_polyfill' so the grouping below works
    gdf_hex.rename(columns={h3_col_name: 'h3_polyfill'}, inplace=True)
    
    # 3. Create Features for Aggregation
    gdf_hex['Day_Num'] = gdf_hex['IncidentDate'].dt.dayofweek 
    
    # 4. Aggregation
    print("  Aggregating counts...")
    grouped = gdf_hex.groupby(
        ['h3_polyfill', 'Day_Num', 'Time_Period'], 
        observed=True
    ).size().reset_index(name='Incident_Count')
    
    # 5. Add Centroid Coordinates
    print("  Calculating centroids...")
    
    # --- FIX START ---
    # Instead of using grouped['h3_polyfill'].h3..., we use the DataFrame accessor.
    
    # A. Set the H3 ID as the index (h3pandas likes this)
    grouped = grouped.set_index('h3_polyfill')
    
    # B. Generate geometry (Points) from the H3 index
    # h3_to_geo() returns a GeoDataFrame with a 'geometry' column
    grouped_geo = grouped.h3.h3_to_geo()
    
    # C. Extract Lat/Lon from the geometry column
    grouped_geo['Latitude'] = grouped_geo.geometry.y
    grouped_geo['Longitude'] = grouped_geo.geometry.x
    
    # D. Reset index to get 'h3_polyfill' back as a normal column
    grouped = grouped_geo.reset_index()
    
    # E. Drop the geometry column (we only need lat/lon for the ML model)
    grouped = grouped.drop(columns=['geometry'])
    # --- FIX END ---
    
    print(f"  Result: {len(grouped)} rows of training data.")
    return grouped

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process 2023 (Training Data)
    df_train = process_year(INPUT_2023, '2023')
    if df_train is not None:
        output_file = f'{OUTPUT_DIR}/train_2023.csv'
        df_train.to_csv(output_file, index=False)
        print(f"Saved training data to {output_file}")
        
    print("-" * 30)
    
    # Process 2024 (Testing Data)
    df_test = process_year(INPUT_2024, '2024')
    if df_test is not None:
        output_file = f'{OUTPUT_DIR}/test_2024.csv'
        df_test.to_csv(output_file, index=False)
        print(f"Saved testing data to {output_file}")