# Data Directory

## Dataset: Synthetic Water Sensor Dataset (BattLeDIM-derived)

### Source
Generated from real sensor statistics extracted from the BattLeDIM 2020 competition dataset
(Zenodo ID: 4017659, https://zenodo.org/record/4017659).

The BattLeDIM dataset provides real SCADA measurements from the L-Town water distribution
network (33 pressure sensors, 3 flow sensors, 1 tank level sensor, 82 AMR demand meters).

### Why Synthetic Generation Was Used
The BattLeDIM leakage label files represent continuous background pipe leakages that persist
for most of the year (e.g., pipe p257 leaks from Jan 8 through Dec 31, 2018). This makes
the dataset unsuitable for binary anomaly detection as-is — the "normal" class would contain
only the first 8 days of data (~2,300 rows), which is insufficient for Transformer training.

The synthetic generation approach:
1. Extracts real sensor means, standard deviations, and ranges from the clean 8-day normal period
2. Generates a full year of realistic sensor data with:
   - Daily demand cycles (morning/evening peaks)
   - Weekly patterns (weekday vs weekend)
   - Seasonal variation
   - Realistic sensor correlations
   - Autocorrelated noise
3. Injects 15 discrete anomaly events representing real water system faults:
   - Pressure drops (pipe bursts)
   - Flow spikes (pump failures)
   - Level anomalies (tank overflow/drain)
   - Combined multi-sensor events

### Dataset Properties
- Rows: 105,120
- Features: 12 sensors + 1 timestamp + 1 label = 14 columns
- Sampling: 5-minute intervals
- Period: 2018-01-01 to 2018-12-31
- Normal: 91,056 rows (86.62%)
- Anomaly: 14,064 rows (13.38%)
- Missing values: 0
- Duplicate rows: 0

### Sensor Columns
| Column  | Type     | Unit   | Description                    |
|---------|----------|--------|--------------------------------|
| n1      | Pressure | m      | Network node 1 pressure        |
| n54     | Pressure | m      | Network node 54 pressure       |
| n105    | Pressure | m      | Network node 105 pressure      |
| n163    | Pressure | m      | Network node 163 pressure      |
| n215    | Pressure | m      | Network node 215 pressure      |
| n332    | Pressure | m      | Network node 332 pressure      |
| n458    | Pressure | m      | Network node 458 pressure      |
| n549    | Pressure | m      | Network node 549 pressure      |
| p227    | Flow     | m³/h   | Pipe 227 flow rate             |
| p235    | Flow     | m³/h   | Pipe 235 flow rate             |
| PUMP_1  | Flow     | m³/h   | Pump 1 flow rate               |
| T1      | Level    | m      | Tank 1 water level             |
| label   | Target   | 0/1    | 0=Normal, 1=Anomaly            |

### Anomaly Events
15 events injected across the year:
- 6 pressure drop events (pipe burst simulation)
- 4 flow spike events (pump failure simulation)
- 2 level anomaly events (tank overflow/drain)
- 3 combined multi-sensor events (severe fault simulation)

### Files
```
data/
├── raw/
│   ├── water_sensor_data.csv      # Main dataset (105,120 rows)
│   ├── anomaly_events.json        # Anomaly event log
│   ├── sensor_stats.json          # Real sensor statistics from BattLeDIM
│   ├── 2018_SCADA.xlsx            # Original BattLeDIM 2018 SCADA data
│   ├── 2019_SCADA.xlsx            # Original BattLeDIM 2019 SCADA data
│   ├── dataset_configuration.yaml # BattLeDIM competition config
│   └── README.txt                 # BattLeDIM original README
└── processed/
    ├── train.csv                  # Training split (Jan–Aug 2018)
    ├── val.csv                    # Validation split (Sep–Oct 2018)
    ├── test.csv                   # Test split (Nov–Dec 2018)
    ├── scaler.pkl                 # Fitted StandardScaler
    └── preprocessing_meta.json    # Preprocessing configuration
```

### Chronological Split
| Split      | Period              | Rows   | Anomaly % |
|------------|---------------------|--------|-----------|
| Train      | Jan 1 – Aug 31      | 69,984 | 12.6%     |
| Validation | Sep 1 – Oct 31      | 17,568 | 9.8%      |
| Test       | Nov 1 – Dec 31      | 17,568 | 20.2%     |

Note: The test set has higher anomaly rate because two large combined events
(Nov 27–Dec 2 and Dec 17–20) fall in this period.
