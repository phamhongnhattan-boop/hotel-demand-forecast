"""Project settings. Change values here instead of inside the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "hotel_bookings.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Hotels in the dataset: "City Hotel" (Lisbon) and "Resort Hotel" (Algarve)
DEFAULT_HOTEL = "City Hotel"

# Forecast this many days ahead (direct strategy: every lag feature is >= HORIZON)
HORIZON = 14

# Time split (data covers arrivals from 2015-07-01 to 2017-08-31)
# The first month is skipped: guests who arrived before 2015-07-01 are missing from the data,
# so the first days show too few rooms sold.
TRAIN_START = "2015-08-01"
VALID_START = "2017-03-01"
TEST_START = "2017-06-01"
DATA_END = "2017-08-31"

RANDOM_STATE = 42
