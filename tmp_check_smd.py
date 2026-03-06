import pandas as pd
import glob
import os

files = glob.glob('c:/Users/Hus96/Projects/jetbrains-predictive-alerting/data/multivariate/*.csv')

total_rows = 0
total_incidents = 0

print(f"{'File':<25} | {'Rows':<6} | {'Incidents':<10} | {'Incident Rate'}")
print("-" * 65)
for f in sorted(files):
    df = pd.read_csv(f)
    rows = len(df)
    incidents = df['label'].sum()
    rate = (incidents / rows) * 100
    
    total_rows += rows
    total_incidents += incidents
    
    basename = os.path.basename(f)
    print(f"{basename:<25} | {rows:<6} | {incidents:<10} | {rate:.2f}%")

print("-" * 65)
print(f"{'TOTAL':<25} | {total_rows:<6} | {total_incidents:<10} | {(total_incidents/total_rows)*100:.2f}%")
