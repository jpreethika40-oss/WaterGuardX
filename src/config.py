"""Project-wide configuration."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_RAW    = ROOT / 'data' / 'raw'
DATA_PROC   = ROOT / 'data' / 'processed'
MODELS_DIR  = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'

RAW_DATA_FILE   = DATA_RAW / 'water_sensor_data.csv'
ANOMALY_LOG     = DATA_RAW / 'anomaly_events.json'
SENSOR_STATS    = DATA_RAW / 'sensor_stats.json'

# ── Sensor columns ────────────────────────────────────────────────────────────
PRESSURE_SENSORS = ['n1', 'n54', 'n105', 'n163', 'n215', 'n332', 'n458', 'n549']
FLOW_SENSORS     = ['p227', 'p235', 'PUMP_1']
LEVEL_SENSORS    = ['T1']
SENSOR_COLS      = PRESSURE_SENSORS + FLOW_SENSORS + LEVEL_SENSORS
TARGET_COL       = 'label'
TIMESTAMP_COL    = 'Timestamp'

# ── Splits (chronological) ────────────────────────────────────────────────────
TRAIN_END = '2018-08-31 23:55:00'   # ~67% of year
VAL_END   = '2018-10-31 23:55:00'   # ~17% of year
# Test: remainder                    # ~16% of year

# ── Sequence windows ──────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 48    # 48 × 5 min = 4 hours of context
STRIDE          = 1     # slide by 1 step

# ── Preprocessing ─────────────────────────────────────────────────────────────
SCALER_TYPE = 'standard'   # 'standard' or 'minmax'

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
