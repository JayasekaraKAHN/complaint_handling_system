import pandas as pd

# Load and analyze solution patterns
df = pd.read_csv('data/datafinal.csv')
valid_df = df.dropna(subset=['Issue_Description', 'Solution'])

print("=== SOLUTION PATTERN ANALYSIS ===")
print(f"Total records: {len(df)}")
print(f"Valid solutions: {len(valid_df)}")

print("\n=== SAMPLE ISSUE-SOLUTION PAIRS ===")
for idx, (i, row) in enumerate(valid_df.head(5).iterrows(), 1):
    print(f"\nRecord {idx}:")
    print(f"Issue: {row['Issue_Description']}")
    print(f"Device: {row.get('Device_type_settings_VPN_APN', 'N/A')}")
    print(f"Site/KPI: {row.get('Site_KPI_Alarm', 'N/A')}")
    print(f"Solution: {row['Solution']}")
    print("-" * 80)

print("\n=== UNIQUE SOLUTIONS ===")
solutions = valid_df['Solution'].unique()
for i, solution in enumerate(solutions[:10], 1):
    print(f"{i}. {solution}")

print(f"\n=== LOCATION DATA ===")
if 'Lon' in df.columns and 'Lat' in df.columns:
    location_data = df[['Lon', 'Lat']].dropna()
    print(f"Records with coordinates: {len(location_data)}")
    if len(location_data) > 0:
        print("Sample coordinates:")
        print(location_data.head(3))
