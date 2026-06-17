# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
ARTIFACTS = PROJECT_ROOT / 'artifacts'

SEED = 42
TRAIN_MAX_STEP = 34  # steps 1 to 34 are train, 35 to 49 are test
N_LOCAL_FEATURES = 93  # first 93 features are local, the remaining 72 are neighbor aggregates

LOCAL_FEATURE_COLS = [f'feat_{i}' for i in range(1, N_LOCAL_FEATURES + 1)]
ALL_FEATURE_COLS = [f'feat_{i}' for i in range(1, 166)]