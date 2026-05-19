# config.py

import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

INP_FILE = INPUT_DIR / "config_a.inp"
PIPE_SPREADSHEET = INPUT_DIR / "network_properties.xlsx"
PHYSICAL_PROPERTY_SPREADSHEET = INPUT_DIR / "physical_properties.xlsx"


# ============================================================
# Model settings
# ============================================================

DURATION = 1 * 3600 # Seconds
TIME_STEP = 5 # Seconds
G = 9.81 # Gravity

F1 = 1.1
F2 = -4
FACTOR = F1 * 10 ** F2  # Axial mixing convection velocity factor

COLD_WATER = 15 # Cold water source temperature (assigned to the reservoir)
HOT_WATER = 70 # Hot water source temperature (right after the heaters/thermomixing valve)

LOW_INTERVAL_TEMP = 20 # Only for line plot in dashboard
HIGH_INTERVAL_TEMP = 30 # Only for line plot in dashboard

READ_INSULATION = True # If False, ignores insulation data from PIPE_SPREADSHEET


# ============================================================
# Demand / household settings
# ============================================================

"""
Notes
-----
- Every demand node must be listed as a dictionary in MY_APPLIANCES with:
  * 'name' (ID of the demand node in the .inp file)
  * 'target_temp' (Any value between COLD_WATER and HOT_WATER)

- 'target_temp' is not required for points connected to just the hot- or just the cold-water lines.
- 'target_temp' represents the desired temperature when the demand point is used.
- Valves called "VC_" + "name" and "VH_" + "name" must exist upstream of node "name" coming
  from the cold-water and hot-water lines, respectively, if 'target_temp' exists
"""
MY_APPLIANCES = [
    {"name": "demand", "target_temp": 50},
    # types do not matter if working with fixed demands
]

RANDOM_PATTERNS = False # True = pySIMDEUM, False = demand patterns form .inp file
NUMBER_OF_PEOPLE = 4 # Only relevant if RANDOM_PATTERNS

if not RANDOM_PATTERNS:
    NUMBER_OF_SIMULATIONS = 1
else:
    NUMBER_OF_SIMULATIONS = 1000


# ============================================================
# Plot / dashboard labels
# ============================================================

LEGEND_1 = "Highest PWA"

RESULTS_TEMPLATE = [
    {
        "name": "max_age",
        "legend": LEGEND_1,
        "show2": True,
        "value": 0,
        "coordinates": {"pipe": None, "time": None},
        "results": None,
        "wn": None,
        "temperatures": None,
        "cti": None,
    },
]

ELEMENT_OPTIONS = ["Link", "Node"]
ELEMENT_DEFAULT = "Link"

VARIABLE_OPTIONS_NODES = ["Age", "Demand"]
VARIABLE_DEFAULT = "Age"

TIME_OPTIONS = ["Weekly", "Daily", "Bi-hourly"]
TIME_DEFAULT = "Weekly"

SIMULATION_OPTIONS = [LEGEND_1]
SIMULATION_DEFAULT = LEGEND_1

DEMAND_UNITS = "L/min"
AGE_UNITS = "hours"


def get_cti_option_text(
        low_interval_temp: float,
        high_interval_temp: float,
        ) -> str:
    """
    Build the dashboard label for consecutive time in a target temperature range.
    """
    return f"Consecutive Time Between {low_interval_temp}°C and {high_interval_temp}°C"


def get_variable_options_links(
        low_interval_temp: float,
        high_interval_temp: float,
        ) -> list[str]:
    """
    Build link-variable options using the configured CTI interval.
    """
    cti_option_text = get_cti_option_text(low_interval_temp, high_interval_temp)
    return ["Age", "Temperature", cti_option_text]


# ============================================================
# Loaders
# ============================================================


def load_physical_property_tables(
        path: str,
        ) -> dict[str, pd.DataFrame]:
    """
    Load all sheets from the physical-properties workbook.

    Parameters
    ----------
    path : str
        Path to the Excel workbook.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary keyed by sheet name.
    """
    return pd.read_excel(path, sheet_name=None)


def build_interpolators(
        physical_property_tables: dict[str, pd.DataFrame],
        ) -> dict[str, interp1d]:
    """
    Build 1D linear interpolators for each physical-property sheet.

    Assumes the first column contains temperature values and the second
    column contains property values.
    """
    interpolators = {}

    for sheet_name, data in physical_property_tables.items():
        temperatures = data.iloc[:, 0].values
        values = data.iloc[:, 1].values

        interpolators[sheet_name] = interp1d(
            temperatures,
            values,
            kind="linear",
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )

    return interpolators


def load_pipes_df(
        path: str,
        ) -> pd.DataFrame:
    """
    Load the 'Pipes' sheet and force the ID column to string.
    """
    pipes_df = pd.read_excel(path, sheet_name="Pipes")
    pipes_df.iloc[:, 0] = pipes_df.iloc[:, 0].astype(str)
    return pipes_df
