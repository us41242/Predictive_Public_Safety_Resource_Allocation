import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
TRAIN_PATH = 'data/ml_ready/train_2023.csv'
TEST_PATH = 'data/ml_ready/test_2024.csv'
FIGURES_DIR = 'reports/figures'

def train_and_evaluate():
    print("Loading datasets...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("Run ml_prep.py first!")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # 1. Preprocessing
    # We need to convert 'Time_Period' (String) into numbers for the machine
    # Mapping: Morning=0, Afternoon=1, etc.
    time_mapping = {label: idx for idx, label in enumerate(train_df['Time_Period'].unique())}
    
    train_df['Time_Code'] = train_df['Time_Period'].map(time_mapping)
    test_df['Time_Code'] = test_df['Time_Period'].map(time_mapping)
    
    # Define Features (X) and Target (y)
    features = ['Latitude', 'Longitude', 'Day_Num', 'Time_Code']
    target = 'Incident_Count'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    # 2. Training
    print("Training Random Forest Regressor (this may take a minute)...")
    # n_estimators=100 means we build 100 decision trees
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 3. Prediction
    print("Predicting on 2024 data...")
    predictions = model.predict(X_test)
    
    # 4. Evaluation
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Interpretation: On average, the model is off by {mae:.2f} incidents per zone.")
    
    # 5. Visualization: Feature Importance
    # What drove the predictions? Location? Time?
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features, y=model.feature_importances_)
    plt.title('Feature Importance (What drives crime risk?)')
    plt.ylabel('Importance Score')
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f'{FIGURES_DIR}/08_feature_importance.png')
    print(f"Saved feature importance plot to {FIGURES_DIR}")

if __name__ == "__main__":
    train_and_evaluate()