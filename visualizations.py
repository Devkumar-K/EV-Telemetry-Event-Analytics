import os
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import seaborn as sns

log = logging.getLogger("charge_analytics")

def generate_all_visualizations(output_dir: Path):
    """
    Reads output datasets and generates visualizations using matplotlib & seaborn.
    Saves plots to output_dir/visualizations/
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Datasets
    ign_file = output_dir / "IgnitionEvents.csv"
    charge_file = output_dir / "ChargingEvents.csv"
    status_file = output_dir / "ChargingStatusEvents.csv"
    
    try:
        if ign_file.exists():
            ign_df = pd.read_csv(ign_file)
            ign_df['event_ts'] = pd.to_datetime(ign_df['event_ts'])
            _plot_ignition_timeline(ign_df, vis_dir)
            # Battery pct is not in IgnitionEvents.csv, so we plot it from ChargingEvents.
        
        if charge_file.exists():
            charge_df = pd.read_csv(charge_file)
            charge_df['start_ts'] = pd.to_datetime(charge_df['start_ts'])
            charge_df['end_ts'] = pd.to_datetime(charge_df['end_ts'])
            _plot_charge_duration_hist(charge_df, vis_dir)
            _plot_battery_gain_boxplot(charge_df, vis_dir)
            _plot_sessions_per_vehicle(charge_df, vis_dir)
            _plot_duration_vs_gain(charge_df, vis_dir)
            _plot_charging_heatmap(charge_df, vis_dir)
            
            # Since we need battery level over time, let's construct it from ChargingEvents start & end battery pct as a proxy if possible.
            _plot_battery_over_time_from_charges(charge_df, vis_dir)
            
    except Exception as e:
        log.error("Failed to generate visualizations: %s", str(e), exc_info=True)


def _plot_ignition_timeline(ign_df: pd.DataFrame, out_dir: Path):
    if ign_df.empty: return
    
    plt.figure(figsize=(12, 6))
    
    # Map events to distinct Y values per vehicle
    vehicles = ign_df['vehicle_id'].unique()
    v_map = {v: i for i, v in enumerate(vehicles)}
    
    ign_on = ign_df[ign_df['event'] == 'ignitionOn']
    ign_off = ign_df[ign_df['event'] == 'ignitionOff']
    
    plt.scatter(ign_on['event_ts'], ign_on['vehicle_id'].map(v_map), 
                c='green', label='Ignition ON', marker='^', alpha=0.7)
    plt.scatter(ign_off['event_ts'], ign_off['vehicle_id'].map(v_map), 
                c='red', label='Ignition OFF', marker='v', alpha=0.7)
    
    plt.yticks(list(v_map.values()), list(v_map.keys()))
    plt.xlabel('Time')
    plt.ylabel('Vehicle ID')
    plt.title('Ignition Events Timeline per Vehicle')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "ignition_timeline.png", dpi=150)
    plt.close()


def _plot_charge_duration_hist(charge_df: pd.DataFrame, out_dir: Path):
    if charge_df.empty: return
    
    plt.figure(figsize=(10, 5))
    sns.histplot(charge_df['duration_sec'] / 60, bins=30, kde=True, color='skyblue')
    plt.xlabel('Duration (Minutes)')
    plt.ylabel('Frequency')
    plt.title('Charging Session Duration Distribution')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "charge_duration_hist.png", dpi=150)
    plt.close()


def _plot_battery_gain_boxplot(charge_df: pd.DataFrame, out_dir: Path):
    if charge_df.empty: return
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=charge_df['delta_battery_pct'], color='lightgreen')
    plt.ylabel('Battery Percentage Gain (%)')
    plt.title('Battery Percentage Change During Charging Sessions')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "battery_gain_boxplot.png", dpi=150)
    plt.close()


def _plot_sessions_per_vehicle(charge_df: pd.DataFrame, out_dir: Path):
    if charge_df.empty: return
    
    plt.figure(figsize=(10, 6))
    session_counts = charge_df['vehicle_id'].value_counts()
    sns.barplot(x=session_counts.index, y=session_counts.values, palette='viridis')
    plt.xlabel('Vehicle ID')
    plt.ylabel('Number of Charging Sessions')
    plt.title('Number of Charging Sessions per Vehicle')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "charge_sessions_per_vehicle.png", dpi=150)
    plt.close()


def _plot_duration_vs_gain(charge_df: pd.DataFrame, out_dir: Path):
    if charge_df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=charge_df['duration_sec'] / 60, y=charge_df['delta_battery_pct'], 
                    hue=charge_df['vehicle_id'], palette='Set2', alpha=0.8)
    plt.xlabel('Duration (Minutes)')
    plt.ylabel('Battery Gain (%)')
    plt.title('Charging Session Duration vs Battery Gain')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "duration_vs_gain_scatter.png", dpi=150)
    plt.close()


def _plot_charging_heatmap(charge_df: pd.DataFrame, out_dir: Path):
    if charge_df.empty: return
    
    # Extract hour of day
    charge_df['hour'] = charge_df['start_ts'].dt.hour
    
    # Count sessions per vehicle per hour
    heat_data = charge_df.groupby(['vehicle_id', 'hour']).size().unstack(fill_value=0)
    
    # Ensure all 24 hours are represented
    for h in range(24):
        if h not in heat_data.columns:
            heat_data[h] = 0
    heat_data = heat_data[range(24)] # order columns 0-23
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heat_data, cmap='YlOrRd', annot=True, fmt="d", cbar_kws={'label': 'Number of Sessions'})
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Vehicle ID')
    plt.title('Charging Activity Heatmap by Hour of Day')
    plt.tight_layout()
    plt.savefig(out_dir / "charging_activity_heatmap.png", dpi=150)
    plt.close()


def _plot_battery_over_time_from_charges(charge_df: pd.DataFrame, out_dir: Path):
    """ Proxy for battery over time using charging session start/end states """
    if charge_df.empty: return
    
    plt.figure(figsize=(14, 6))
    
    # Create a sparse timeline of battery levels at start and end of charges
    time_points = []
    for _, row in charge_df.iterrows():
        time_points.append({'vehicle_id': row['vehicle_id'], 'ts': row['start_ts'], 'battery': row['start_battery_pct']})
        time_points.append({'vehicle_id': row['vehicle_id'], 'ts': row['end_ts'], 'battery': row['end_battery_pct']})
        
    df_batt = pd.DataFrame(time_points).sort_values(by='ts')
    
    sns.lineplot(data=df_batt, x='ts', y='battery', hue='vehicle_id', marker='o', palette='tab10')
    
    plt.xlabel('Time')
    plt.ylabel('Battery Percentage (%)')
    plt.title('Battery Level Over Time (During Valid Charging Sessions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "battery_over_time.png", dpi=150)
    plt.close()
