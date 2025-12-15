import pandas as pd

file_path = 'data/cleaned_lvmpd_incidents.parquet'
df = pd.read_parquet(file_path)

# Analyze 'Miscellaneous' Crime Category 
misc_df = df[df['Crime_Category'] == 'Miscellaneous']

# Top 30 most frequent IncidentTypeDescription in 'Miscellaneous'
top_missed_crimes = misc_df['IncidentTypeDescription'].value_counts().head(30)

print(f"Total 'Miscellaneous' Incidents: {len(misc_df)}")
print("\n--- Top 30 Descriptions trapped in Miscellaneous ---")
print(top_missed_crimes)