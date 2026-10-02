"""
Generate a realistic synthetic water sensor dataset.

Approach:
- Learn the statistical properties of BattLeDIM 2018 SCADA data
  (pressure, flow, level sensors) from the clean normal period
- Generate a full year of synthetic sensor data with realistic:
  * daily patterns (demand cycles)
  * weekly patterns (weekday vs weekend)
  * seasonal variation
  * sensor correlations
  * realistic noise
- Inject discrete anomaly events (10-15% of timesteps) representing:
  * pressure drops (pipe burst)
  * flow spikes (pump failure)
  * level anomalies (tank overflow/drain)
  * combined multi-sensor events
- Result: ~105,120 rows, 37 features, binary label, 5-min resolution

This is documented as synthetic data generated from real sensor statistics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

# ── 1. Load real BattLeDIM statistics from XLSX ──────────────────────────────
print("Loading real BattLeDIM sensor statistics...")
xl_2018 = pd.ExcelFile('data/raw/2018_SCADA.xlsx')

df_press = pd.read_excel('data/raw/2018_SCADA.xlsx', sheet_name='Pressures (m)')
df_flow  = pd.read_excel('data/raw/2018_SCADA.xlsx', sheet_name='Flows (m3_h)')
df_level = pd.read_excel('data/raw/2018_SCADA.xlsx', sheet_name='Levels (m)')

# Use only the clean period (first 8 days = 2310 rows) for statistics
clean_mask = df_press['Timestamp'] < '2018-01-09'
df_press_clean = df_press[clean_mask].copy()
df_flow_clean  = df_flow[clean_mask].copy()
df_level_clean = df_level[clean_mask].copy()

print(f"Clean period rows: {clean_mask.sum()}")

# Select a representative subset of pressure sensors (avoid redundancy)
# Use 8 pressure sensors that cover different network zones
pressure_sensors = ['n1', 'n54', 'n105', 'n163', 'n215', 'n332', 'n458', 'n549']
flow_sensors     = ['p227', 'p235', 'PUMP_1']
level_sensors    = ['T1']

all_sensors = pressure_sensors + flow_sensors + level_sensors
print(f"Selected sensors: {all_sensors}")

# Compute statistics from clean period
stats = {}
for col in pressure_sensors:
    stats[col] = {
        'mean': df_press_clean[col].mean(),
        'std':  df_press_clean[col].std(),
        'min':  df_press_clean[col].min(),
        'max':  df_press_clean[col].max(),
    }
for col in flow_sensors:
    stats[col] = {
        'mean': df_flow_clean[col].mean(),
        'std':  df_flow_clean[col].std(),
        'min':  df_flow_clean[col].min(),
        'max':  df_flow_clean[col].max(),
    }
for col in level_sensors:
    stats[col] = {
        'mean': df_level_clean[col].mean(),
        'std':  df_level_clean[col].std(),
        'min':  df_level_clean[col].min(),
        'max':  df_level_clean[col].max(),
    }

print("\nSensor statistics from clean period:")
for k, v in stats.items():
    print(f"  {k}: mean={v['mean']:.2f}, std={v['std']:.2f}, "
          f"min={v['min']:.2f}, max={v['max']:.2f}")

# ── 2. Generate timestamps ────────────────────────────────────────────────────
n_rows = 105120  # full year at 5-min resolution
timestamps = pd.date_range('2018-01-01 00:00:00', periods=n_rows, freq='5min')

# Time features for pattern generation
hour_of_day  = timestamps.hour + timestamps.minute / 60.0
day_of_week  = timestamps.dayofweek   # 0=Mon, 6=Sun
day_of_year  = timestamps.dayofyear

# ── 3. Generate demand pattern ────────────────────────────────────────────────
# Daily demand cycle: low at night, peak morning/evening
def demand_pattern(hour):
    """Normalized demand [0,1] based on hour of day."""
    # Morning peak ~7-9, evening peak ~18-20, low at 2-5
    morning = np.exp(-0.5 * ((hour - 8.0) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2)
    night   = 0.15
    return night + 0.6 * morning + 0.5 * evening

demand = demand_pattern(hour_of_day.values)

# Weekend reduction
weekend_factor = np.where(day_of_week.values >= 5, 0.75, 1.0)
demand = demand * weekend_factor

# Seasonal variation (higher demand in summer)
seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year.values - 80) / 365)
demand = demand * seasonal

# Normalize to [0, 1]
demand = (demand - demand.min()) / (demand.max() - demand.min())

# ── 4. Generate sensor readings ───────────────────────────────────────────────
def gen_sensor(name, demand, noise_scale=0.3):
    """Generate sensor time series from demand pattern + noise."""
    s = stats[name]
    # Pressure: inversely related to demand (higher demand → lower pressure)
    if name.startswith('n'):
        base = s['mean'] - (demand - 0.5) * s['std'] * 1.5
    # Flow: positively related to demand
    elif name.startswith('p') or name == 'PUMP_1':
        base = s['mean'] + (demand - 0.5) * s['std'] * 2.0
    # Level: slowly varying tank level
    else:
        # Tank level oscillates with demand (fills at night, drains during day)
        base = s['mean'] + 0.3 * np.sin(2 * np.pi * hour_of_day.values / 24) * s['std']
    
    # Add realistic noise
    noise = np.random.normal(0, s['std'] * noise_scale, n_rows)
    # Add autocorrelation (smooth noise)
    for i in range(1, n_rows):
        noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
    
    signal = base + noise
    # Clip to realistic range (allow slight exceedance for anomalies)
    signal = np.clip(signal, s['min'] * 0.7, s['max'] * 1.3)
    return signal

print("\nGenerating normal sensor signals...")
sensor_data = {}
for name in all_sensors:
    sensor_data[name] = gen_sensor(name, demand)

# ── 5. Inject anomaly events ──────────────────────────────────────────────────
print("Injecting anomaly events...")

labels = np.zeros(n_rows, dtype=int)
anomaly_log = []

# Target: ~12% anomaly rate (realistic for water system)
# Inject 15 distinct anomaly events — durations chosen to reach ~12% total

anomaly_events = [
    # (start_day, duration_hours, type, affected_sensors, magnitude)
    # Pressure drops (pipe burst) — longer durations for realistic pipe events
    (45,  72,  'pressure_drop',  ['n1', 'n54'],            -0.25),
    (78,  96,  'pressure_drop',  ['n105', 'n163'],          -0.30),
    (120, 48,  'pressure_drop',  ['n215', 'n332'],          -0.20),
    (180, 120, 'pressure_drop',  ['n458', 'n549'],          -0.35),
    (220, 80,  'pressure_drop',  ['n1', 'n54', 'n105'],     -0.28),
    (290, 64,  'pressure_drop',  ['n163', 'n215'],          -0.22),
    # Flow spikes (pump failure / valve issue)
    (55,  36,  'flow_spike',     ['p227', 'p235'],           0.40),
    (140, 48,  'flow_spike',     ['PUMP_1'],                 0.50),
    (200, 40,  'flow_spike',     ['p227', 'PUMP_1'],         0.45),
    (310, 56,  'flow_spike',     ['p235', 'PUMP_1'],         0.35),
    # Level anomalies (tank overflow/drain)
    (95,  72,  'level_anomaly',  ['T1'],                     0.30),
    (250, 80,  'level_anomaly',  ['T1'],                    -0.25),
    # Combined multi-sensor events (most severe)
    (160, 120, 'combined',       ['n1', 'n54', 'p227', 'T1'],         -0.30),
    (330, 144, 'combined',       ['n105', 'n163', 'PUMP_1', 'T1'],    -0.25),
    (350, 96,  'combined',       ['n215', 'n332', 'p235', 'T1'],      -0.20),
]

for start_day, dur_hours, atype, affected, magnitude in anomaly_events:
    # Convert to row indices
    start_idx = start_day * 24 * 12  # 12 rows per hour (5-min)
    end_idx   = min(start_idx + int(dur_hours * 12), n_rows)
    
    # Mark as anomaly
    labels[start_idx:end_idx] = 1
    
    # Apply sensor perturbation
    for sensor in affected:
        if sensor in sensor_data:
            s = stats[sensor]
            # Gradual onset (ramp up over first 20% of event)
            ramp_len = max(1, int((end_idx - start_idx) * 0.2))
            ramp = np.linspace(0, 1, ramp_len)
            
            perturbation = np.ones(end_idx - start_idx) * magnitude * s['std'] * 3
            perturbation[:ramp_len] *= ramp
            
            # Add anomaly-specific noise
            anom_noise = np.random.normal(0, s['std'] * 0.5, end_idx - start_idx)
            sensor_data[sensor][start_idx:end_idx] += perturbation + anom_noise
            
            # Clip to prevent unrealistic values
            sensor_data[sensor] = np.clip(
                sensor_data[sensor],
                s['min'] * 0.5,
                s['max'] * 1.5
            )
    
    anomaly_log.append({
        'start': str(timestamps[start_idx]),
        'end':   str(timestamps[end_idx-1]),
        'type':  atype,
        'sensors': affected,
        'duration_hours': dur_hours,
        'rows': end_idx - start_idx,
    })

# ── 6. Build final DataFrame ──────────────────────────────────────────────────
print("Building final DataFrame...")
df = pd.DataFrame({'Timestamp': timestamps})
for name in all_sensors:
    df[name] = sensor_data[name]
df['label'] = labels

# Round to 2 decimal places (realistic sensor precision)
for col in all_sensors:
    df[col] = df[col].round(2)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SYNTHETIC DATASET SUMMARY")
print(f"{'='*60}")
print(f"Shape: {df.shape}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Sampling: 5-minute intervals")
print(f"Sensors: {len(all_sensors)}")
print(f"  Pressure: {pressure_sensors}")
print(f"  Flow: {flow_sensors}")
print(f"  Level: {level_sensors}")
print(f"\nClass distribution:")
vc = df['label'].value_counts()
print(f"  Normal (0): {vc[0]} ({vc[0]/len(df)*100:.2f}%)")
print(f"  Anomaly (1): {vc[1]} ({vc[1]/len(df)*100:.2f}%)")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"\nAnomalous events injected: {len(anomaly_events)}")
for ev in anomaly_log:
    print(f"  {ev['type']:20s} {ev['start'][:16]} -> {ev['end'][:16]} "
          f"({ev['duration_hours']}h, sensors: {ev['sensors']})")

# ── 8. Save ───────────────────────────────────────────────────────────────────
Path('data/raw').mkdir(parents=True, exist_ok=True)
df.to_csv('data/raw/water_sensor_data.csv', index=False)
print(f"\nSaved: data/raw/water_sensor_data.csv ({df.shape[0]} rows, {df.shape[1]} cols)")

# Save anomaly log
with open('data/raw/anomaly_events.json', 'w') as f:
    json.dump(anomaly_log, f, indent=2)
print("Saved: data/raw/anomaly_events.json")

# Save sensor statistics
with open('data/raw/sensor_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved: data/raw/sensor_stats.json")
