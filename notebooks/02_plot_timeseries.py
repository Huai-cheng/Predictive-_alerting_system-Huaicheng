import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import download_nab_dataset, generate_labels

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
DATA_PATH = "data/ec2_cpu_utilization_24ae8d.csv"

def plot_timeseries():
    # 1. Fetch Data
    df = download_nab_dataset(DATA_URL, DATA_PATH)
    
    # 2. Generate Ground-Truth Labels dynamically to show incidents
    df = generate_labels(df, window_size=288, z_threshold=3.0)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot the CPU value
    plt.plot(df['timestamp'], df['value'], label='CPU Utilization', color='blue', alpha=0.6)
    
    # Overlay the incidents as red 'x' marks
    incidents = df[df['label'] == 1]
    plt.scatter(incidents['timestamp'], incidents['value'], color='red', marker='x', s=100, label='Incidents (Spikes)')
    
    plt.title('AWS EC2 CPU Utilization Over Time (with anomalies highlighted)')
    plt.xlabel('Time')
    plt.ylabel('CPU Utilization (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on a 5-day window specifically so we can see the spike better
    # Let's take a slice of the dataset
    
    os.makedirs('notebooks', exist_ok=True)
    plot_path = "notebooks/timeseries_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved Time Series plot to {plot_path}")
    
    # Second chart: Zoomed in on an incident
    if len(incidents) > 0:
        first_incident_idx = df.index.get_loc(incidents.index[0])
        start_idx = max(0, first_incident_idx - 100)
        end_idx = min(len(df), first_incident_idx + 100)
        zoom_df = df.iloc[start_idx:end_idx]
        zoom_incidents = zoom_df[zoom_df['label'] == 1]
        
        plt.figure(figsize=(12, 5))
        plt.plot(zoom_df['timestamp'], zoom_df['value'], label='CPU Utils (Zoom)', color='blue', alpha=0.7)
        plt.scatter(zoom_incidents['timestamp'], zoom_incidents['value'], color='red', marker='x', s=100, label='Incidents')
        plt.title('Zoomed-in View of Incident')
        plt.xlabel('Time')
        plt.ylabel('CPU Util (%)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('notebooks/timeseries_zoom_plot.png')
        print("Saved Zoomed Time Series plot to notebooks/timeseries_zoom_plot.png")

if __name__ == "__main__":
    plot_timeseries()
