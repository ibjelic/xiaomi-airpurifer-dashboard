from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import threading
import time
from functools import wraps
from collections import deque

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, send_file

# Import miio with better error handling for Python 3.14 compatibility
AirPurifierMiot = None
OperationMode = None
try:
    from miio import AirPurifierMiot
    # Try to import OperationMode from the integrations module
    try:
        from miio.integrations.zhimi.airpurifier.airpurifier_miot import OperationMode
    except ImportError:
        try:
            from miio import OperationMode
        except ImportError:
            OperationMode = None
except (ImportError, RuntimeError) as e:
    try:
        from miio.airpurifier import AirPurifierMiot
        try:
            from miio.airpurifier import OperationMode
        except ImportError:
            OperationMode = None
    except (ImportError, RuntimeError):
        try:
            from miio.airpurifier_miot import AirPurifierMiot
            try:
                from miio.airpurifier_miot import OperationMode
            except ImportError:
                # Try the full path for OperationMode
                try:
                    from miio.integrations.zhimi.airpurifier.airpurifier_miot import OperationMode
                except ImportError:
                    OperationMode = None
        except (ImportError, RuntimeError):
            import sys
            print("ERROR: Failed to import python-miio. This is likely due to Python 3.14 compatibility issues.")
            print("SOLUTION: Please use Python 3.11 or 3.12 instead.")
            print(f"Current Python version: {sys.version}")
            print("\nTo fix this:")
            print("1. Install Python 3.11 or 3.12")
            print("2. Create a new virtual environment: python3.11 -m venv venv")
            print("3. Activate it and install requirements again")
            sys.exit(1)

# Load environment variables
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("xiaomi_dashboard")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return default


class PurifierClient:
    def __init__(self, ip: str, token: str, level_min: int, level_max: int) -> None:
        if AirPurifierMiot is None:
            raise RuntimeError("AirPurifierMiot not available. Check Python version compatibility.")
        self.dev = AirPurifierMiot(ip, token)
        self.level_min = level_min
        self.level_max = level_max
        self.last_level = None
        
        # Try to get OperationMode from the device's module if not already imported
        global OperationMode
        if OperationMode is None:
            try:
                # Get the module where the device class is defined
                device_module = self.dev.__class__.__module__
                if device_module:
                    import importlib
                    mod = importlib.import_module(device_module)
                    if hasattr(mod, 'OperationMode'):
                        OperationMode = mod.OperationMode
                        print(f"Successfully imported OperationMode from {device_module}")
                    else:
                        # Try common alternative paths
                        try:
                            from miio.integrations.zhimi.airpurifier.airpurifier_miot import OperationMode as OM
                            OperationMode = OM
                            print("Successfully imported OperationMode from airpurifier_miot")
                        except ImportError:
                            print(f"Warning: OperationMode not found in {device_module}")
            except Exception as e:
                print(f"Error importing OperationMode: {e}")
                import traceback
                traceback.print_exc()

    def status(self) -> dict:
        try:
            st = self.dev.status()
            mode = getattr(st, "mode", None)
            # Convert enum to string if it's an enum
            if mode is not None:
                try:
                    mode = str(mode.value) if hasattr(mode, 'value') else str(mode)
                except (AttributeError, ValueError):
                    mode = str(mode)
            return {
                "aqi": getattr(st, "aqi", None),
                "favorite_level": getattr(st, "favorite_level", None),
                "mode": mode,
                "is_on": getattr(st, "is_on", None),
                "temperature": getattr(st, "temperature", None),
                "humidity": getattr(st, "humidity", None),
            }
        except Exception as e:
            return {"error": str(e)}

    def percent_to_level(self, percent: int) -> int:
        percent = max(0, min(100, percent))
        level = int(round((percent / 100.0) * self.level_max))
        if level < self.level_min:
            return self.level_min
        return min(self.level_max, level)

    def set_mode(self, mode: str) -> bool:
        """Set the purifier mode: auto, manual, or favorite"""
        logger.debug("set_mode entry: mode=%s mode_type=%s", mode, type(mode))
        try:
            if hasattr(self.dev, "on"):
                self.dev.on()
            
            logger.debug(
                "Before set_mode call: mode=%s has_OperationMode=%s dev_set_mode_type=%s",
                mode,
                OperationMode is not None,
                type(self.dev.set_mode),
            )
            
            # Use OperationMode enum - it's required by the library
            if OperationMode is not None:
                if mode == "auto":
                    mode_to_set = OperationMode.Auto
                elif mode in ["manual", "favorite"]:
                    # Both manual and favorite use the favorite mode on the device
                    mode_to_set = OperationMode.Favorite
                else:
                    mode_to_set = OperationMode.Auto  # Default to Auto
                
                self.dev.set_mode(mode_to_set)
            else:
                raise RuntimeError("OperationMode enum not available. Cannot set mode without enum.")
            
            return True
        except Exception as e:
            # Extract clean error message
            error_msg = str(e)
            # Remove any problematic attributes that might cause serialization issues
            if hasattr(e, '__dict__'):
                error_msg = f"{type(e).__name__}: {error_msg}"
            print(f"Error setting mode: {error_msg}")
            import traceback
            traceback.print_exc()
            return False

    def apply_percent(self, percent: int) -> int:
        """Apply a fan speed percentage by setting favorite level"""
        level = self.percent_to_level(percent)
        if self.last_level == level:
            return level
        try:
            # Ensure device is on
            if hasattr(self.dev, "on"):
                self.dev.on()
            # Set to favorite mode first
            try:
                logger.debug("apply_percent set_mode: has_OperationMode=%s", OperationMode is not None)
                if OperationMode is not None:
                    self.dev.set_mode(OperationMode.Favorite)
                else:
                    raise RuntimeError("OperationMode enum not available. Cannot set mode.")
            except Exception as mode_err:
                # If set_mode fails, try to continue anyway
                logger.warning("set_mode failed in apply_percent: %s", mode_err)
                print(f"Warning: Could not set mode to favorite: {mode_err}")
            # Set the favorite level
            self.dev.set_favorite_level(level)
            self.last_level = level
            return level
        except Exception as e:
            print(f"Error applying percent: {e}")
            import traceback
            traceback.print_exc()
            return self.last_level or level


# Configuration from environment
MIIO_IP = os.getenv("MIIO_IP", "").strip()
MIIO_TOKEN = os.getenv("MIIO_TOKEN", "").strip()
LEVEL_MIN = env_int("LEVEL_MIN", 1)
LEVEL_MAX = env_int("LEVEL_MAX", 14)
POLL_INTERVAL = env_int("POLL_INTERVAL", 5)
PORT = env_int("PORT", 5000)

# Air quality location configuration
AIR_QUALITY_CITY = os.getenv("AIR_QUALITY_CITY", "Belgrade").strip()
AIR_QUALITY_LATITUDE = env_float("AIR_QUALITY_LATITUDE", 44.7866)
AIR_QUALITY_LONGITUDE = env_float("AIR_QUALITY_LONGITUDE", 20.4489)
AIR_QUALITY_TIMEZONE = os.getenv("AIR_QUALITY_TIMEZONE", "Europe/Belgrade").strip()

# Logging configuration
MIN_LOG_INTERVAL = env_int("MIN_LOG_INTERVAL", 10)
MAX_ROWS_PER_FILE = env_int("MAX_ROWS_PER_FILE", 600000)
LOG_ENABLED_DEFAULT = os.getenv("LOG_ENABLED_DEFAULT", "false").lower() == "true"
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

if not MIIO_IP or not MIIO_TOKEN:
    raise SystemExit("MIIO_IP and MIIO_TOKEN are required in .env file")

client = PurifierClient(MIIO_IP, MIIO_TOKEN, LEVEL_MIN, LEVEL_MAX)

# Historical data storage (last 6 hours, 1 reading per minute = 360 points max)
historical_data = deque(maxlen=360)
historical_lock = threading.Lock()

# Weather cache
weather_cache = {}
weather_cache_time = {}
weather_lock = threading.Lock()
WEATHER_CACHE_DURATION = 300  # 5 minutes

# Sleep schedule settings
sleep_schedule = {
    "enabled": False,
    "start_hour": 22,  # 10 PM
    "end_hour": 7,     # 7 AM
    "post_sleep_mode": "auto"  # Mode to return to after sleep time
}

# State persistence file (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "state.json")
SCHEDULE_FILE = os.path.join(SCRIPT_DIR, "schedule.json")

def load_state():
    """Load saved state from file"""
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                saved = json.load(f)
                with state_lock:
                    if "mode" in saved:
                        state["mode"] = saved["mode"]
                    if "auto_max_percent" in saved:
                        state["auto_max_percent"] = saved["auto_max_percent"]
                    if "auto_curve_type" in saved:
                        state["auto_curve_type"] = saved["auto_curve_type"]
                    if "constant_percent" in saved:
                        state["constant_percent"] = saved["constant_percent"]
                    if "current_curve" in saved:
                        state["current_curve"] = saved["current_curve"]
                    if "current_curve_name" in saved:
                        state["current_curve_name"] = saved["current_curve_name"]
                    if "curve_smooth" in saved:
                        state["curve_smooth"] = saved["curve_smooth"]
                    if "selected_curve_name" in saved:
                        state["selected_curve_name"] = saved["selected_curve_name"]
                    if "logging_enabled" in saved:
                        state["logging_enabled"] = saved["logging_enabled"]
                    if "log_interval" in saved:
                        state["log_interval"] = saved["log_interval"]
                    if "temp_assist_enabled" in saved:
                        state["temp_assist_enabled"] = saved["temp_assist_enabled"]
                    if "temp_assist_threshold_c" in saved:
                        state["temp_assist_threshold_c"] = saved["temp_assist_threshold_c"]
                    if "temp_assist_reduce_percent" in saved:
                        state["temp_assist_reduce_percent"] = saved["temp_assist_reduce_percent"]
    except Exception as e:
        print(f"Error loading state: {e}")

def save_state():
    """Save current state to file"""
    try:
        with state_lock:
            to_save = {
                "mode": state["mode"],
                "auto_max_percent": state["auto_max_percent"],
                "auto_curve_type": state["auto_curve_type"],
                "constant_percent": state["constant_percent"],
                "current_curve": state["current_curve"],
                "current_curve_name": state["current_curve_name"],
                "selected_curve_name": state.get("selected_curve_name", "Silent"),
                "curve_smooth": state["curve_smooth"],
                "logging_enabled": state.get("logging_enabled", LOG_ENABLED_DEFAULT),
                "log_interval": state.get("log_interval", MIN_LOG_INTERVAL),
                "temp_assist_enabled": state.get("temp_assist_enabled", False),
                "temp_assist_threshold_c": state.get("temp_assist_threshold_c", 18.0),
                "temp_assist_reduce_percent": state.get("temp_assist_reduce_percent", 10),
            }
        with open(STATE_FILE, 'w') as f:
            json.dump(to_save, f)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_schedule():
    """Load sleep schedule from file"""
    global sleep_schedule
    try:
        if os.path.exists(SCHEDULE_FILE):
            with open(SCHEDULE_FILE, 'r') as f:
                saved = json.load(f)
                sleep_schedule.update(saved)
    except Exception as e:
        print(f"Error loading schedule: {e}")

def save_schedule():
    """Save sleep schedule to file"""
    try:
        with open(SCHEDULE_FILE, 'w') as f:
            json.dump(sleep_schedule, f)
    except Exception as e:
        print(f"Error saving schedule: {e}")

state_lock = threading.Lock()
state = {
    "mode": "auto",  # auto, curve, constant, sleep
    "auto_max_percent": 100,  # MAX% for auto mode
    "auto_curve_type": "linear",  # linear, exponential, logarithmic, quadratic
    "constant_percent": 50,  # Fixed % for constant mode
    "current_curve": [[0, 10], [50, 30], [100, 50], [150, 70], [300, 100]],  # [[aqi, percent], ...]
    "current_curve_name": "Default",  # Name of current curve
    "selected_curve_name": "Silent",  # Selected curve for curve mode
    "curve_smooth": "smooth",  # none, smooth, bezier, catmull-rom
    "last_aqi": None,
    "last_temperature": None,
    "last_humidity": None,
    "current_percent": None,  # Current power level in %
    "last_target_percent": None,
    "last_applied_level": None,
    "last_error": None,
    "last_status": {},
    "last_update": None,
    "last_mode_change": None,
    "mode_change_in_progress": False,
    "logging_enabled": LOG_ENABLED_DEFAULT,
    "log_interval": MIN_LOG_INTERVAL,
    "temp_assist_enabled": False,
    "temp_assist_threshold_c": 18.0,
    "temp_assist_reduce_percent": 10,
}

# Load saved state on startup
load_state()
load_schedule()

# Store saved curves (shared across all users)
saved_curves = {}  # {name: {"points": [[aqi, percent], ...], "smooth": "smooth", "created": timestamp}}
curves_lock = threading.Lock()

# Predefined curves
def create_predefined_curves():
    """Create predefined curve presets"""
    global saved_curves
    with curves_lock:
        # Silent curve: max 60% at 600 AQI, curved around 30-50% from 100+ AQI
        if "Silent" not in saved_curves:
            saved_curves["Silent"] = {
                "points": [[0, 10], [50, 20], [100, 30], [150, 40], [200, 45], [300, 50], [400, 55], [600, 60]],
                "smooth": "smooth",
                "created": dt.datetime.now().isoformat(),
                "preset": True
            }
        
        # Extra Fast and Loud: aggressive curve for fast cleaning
        if "Extra Fast and Loud" not in saved_curves:
            saved_curves["Extra Fast and Loud"] = {
                "points": [[0, 30], [25, 50], [50, 70], [75, 85], [100, 95], [150, 100], [200, 100], [300, 100], [600, 100]],
                "smooth": "smooth",
                "created": dt.datetime.now().isoformat(),
                "preset": True
            }

# Initialize predefined curves
create_predefined_curves()

# Selected curve for curve mode
selected_curve_name = "Silent"  # Default to Silent


class LogManager:
    """Thread-safe CSV logging manager with file rotation"""

    VARIABLE_TYPES = ["aqi", "temperature", "humidity", "fan_speed", "outside_aqi", "outside_temp"]

    def __init__(self, logs_dir: str, max_rows: int = 600000):
        self.logs_dir = logs_dir
        self.max_rows = max_rows
        self.locks = {vtype: threading.Lock() for vtype in self.VARIABLE_TYPES}
        self.row_counts = {vtype: 0 for vtype in self.VARIABLE_TYPES}
        self.current_files = {vtype: None for vtype in self.VARIABLE_TYPES}
        # Note: directories are created on-demand when logging starts

    def _ensure_directories(self):
        """Create log directories if they don't exist"""
        for vtype in self.VARIABLE_TYPES:
            dir_path = os.path.join(self.logs_dir, vtype)
            os.makedirs(dir_path, exist_ok=True)

    def _get_current_file(self, variable_type: str) -> str:
        """Get or create the current log file for a variable type"""
        dir_path = os.path.join(self.logs_dir, variable_type)

        # Find existing files
        existing_files = sorted([
            f for f in os.listdir(dir_path)
            if f.startswith(f"{variable_type}_") and f.endswith(".csv")
        ])

        if existing_files:
            latest_file = os.path.join(dir_path, existing_files[-1])
            # Count rows in the latest file
            try:
                with open(latest_file, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
                if row_count < self.max_rows:
                    self.row_counts[variable_type] = row_count
                    return latest_file
            except:
                pass

        # Create new file
        return self._create_new_file(variable_type)

    def _create_new_file(self, variable_type: str) -> str:
        """Create a new log file with header"""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{variable_type}_{timestamp}.csv"
        filepath = os.path.join(self.logs_dir, variable_type, filename)

        with open(filepath, 'w') as f:
            f.write("timestamp,value\n")

        self.row_counts[variable_type] = 0
        self.current_files[variable_type] = filepath
        return filepath

    def log_value(self, variable_type: str, value: float | int | None):
        """Log a single value to the appropriate CSV file"""
        if variable_type not in self.VARIABLE_TYPES:
            return False
        if value is None:
            return False

        with self.locks[variable_type]:
            try:
                # Ensure directories exist
                self._ensure_directories()

                # Get current file or create new one
                if self.current_files[variable_type] is None:
                    self.current_files[variable_type] = self._get_current_file(variable_type)

                filepath = self.current_files[variable_type]

                # Check if rotation needed
                if self.row_counts[variable_type] >= self.max_rows:
                    filepath = self._create_new_file(variable_type)

                # Write the log entry
                timestamp = dt.datetime.now().isoformat()
                with open(filepath, 'a') as f:
                    f.write(f"{timestamp},{value}\n")

                self.row_counts[variable_type] += 1
                return True
            except Exception as e:
                logger.error(f"Error logging {variable_type}: {e}")
                return False

    def get_log_tree(self) -> dict:
        """Get the file tree structure of all log files"""
        tree = {}

        if not os.path.exists(self.logs_dir):
            return tree

        for vtype in self.VARIABLE_TYPES:
            dir_path = os.path.join(self.logs_dir, vtype)
            if os.path.exists(dir_path):
                files = []
                for filename in sorted(os.listdir(dir_path)):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(dir_path, filename)
                        try:
                            stat = os.stat(filepath)
                            with open(filepath, 'r') as f:
                                row_count = sum(1 for _ in f) - 1  # Subtract header
                            files.append({
                                "name": filename,
                                "path": f"{vtype}/{filename}",
                                "size": stat.st_size,
                                "rows": max(0, row_count),
                                "modified": dt.datetime.fromtimestamp(stat.st_mtime).isoformat()
                            })
                        except:
                            continue
                if files:
                    tree[vtype] = files

        return tree

    def read_file_preview(self, relative_path: str, lines: int = 50) -> dict:
        """Read a preview of a log file"""
        filepath = os.path.join(self.logs_dir, relative_path)

        if not os.path.exists(filepath) or not filepath.endswith('.csv'):
            return {"error": "File not found"}

        try:
            with open(filepath, 'r') as f:
                all_lines = f.readlines()

            total_rows = len(all_lines) - 1  # Subtract header
            header = all_lines[0].strip() if all_lines else ""

            # Get last N lines (most recent data)
            preview_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines[1:]

            return {
                "header": header,
                "preview": [line.strip() for line in preview_lines],
                "total_rows": total_rows,
                "showing": len(preview_lines)
            }
        except Exception as e:
            return {"error": str(e)}

    def get_file_path(self, relative_path: str) -> str | None:
        """Get absolute path to a log file for download"""
        filepath = os.path.join(self.logs_dir, relative_path)
        if os.path.exists(filepath) and filepath.endswith('.csv'):
            return filepath
        return None

    def read_logs_for_analysis(self, variable_type: str, hours: int = 48) -> list:
        """Read log data for analysis within the specified time range"""
        if variable_type not in self.VARIABLE_TYPES:
            return []

        dir_path = os.path.join(self.logs_dir, variable_type)
        if not os.path.exists(dir_path):
            return []

        cutoff = dt.datetime.now() - dt.timedelta(hours=hours)
        data = []

        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith('.csv'):
                continue

            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                timestamp = dt.datetime.fromisoformat(parts[0])
                                if timestamp >= cutoff:
                                    value = float(parts[1])
                                    data.append({
                                        "timestamp": timestamp,
                                        "value": value
                                    })
                            except (ValueError, TypeError):
                                continue
            except:
                continue

        return sorted(data, key=lambda x: x["timestamp"])

    def get_data_range(self) -> dict:
        """Get information about available data ranges"""
        result = {}

        for vtype in self.VARIABLE_TYPES:
            dir_path = os.path.join(self.logs_dir, vtype)
            if not os.path.exists(dir_path):
                result[vtype] = {"hours": 0, "rows": 0}
                continue

            oldest = None
            newest = None
            total_rows = 0

            for filename in os.listdir(dir_path):
                if not filename.endswith('.csv'):
                    continue

                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()

                    if len(lines) < 2:
                        continue

                    total_rows += len(lines) - 1

                    # Get first and last timestamps
                    first_line = lines[1].strip().split(',')
                    last_line = lines[-1].strip().split(',')

                    if first_line:
                        try:
                            ts = dt.datetime.fromisoformat(first_line[0])
                            if oldest is None or ts < oldest:
                                oldest = ts
                        except:
                            pass

                    if last_line:
                        try:
                            ts = dt.datetime.fromisoformat(last_line[0])
                            if newest is None or ts > newest:
                                newest = ts
                        except:
                            pass
                except:
                    continue

            hours = 0
            if oldest and newest:
                hours = (newest - oldest).total_seconds() / 3600

            result[vtype] = {
                "hours": round(hours, 1),
                "rows": total_rows,
                "oldest": oldest.isoformat() if oldest else None,
                "newest": newest.isoformat() if newest else None
            }

        return result


class AnalysisEngine:
    """Engine for generating optimized curves from historical log data"""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager

    def get_available_data_range(self) -> dict:
        """Check how much logged data exists"""
        return self.log_manager.get_data_range()

    def generate_curve(self, hours: int = 48, include_outside_aqi: bool = False,
                      max_cutoff: float = 1.0, weekdays: list = None) -> dict:
        """Generate an optimized AQI->fan_speed curve from historical data

        Args:
            hours: Number of hours of data to analyze
            include_outside_aqi: Whether to factor in outside AQI
            max_cutoff: Maximum percentile cutoff (0.2-1.0 for 20%-100%)
            weekdays: List of weekdays to include (0=Monday, 6=Sunday), None for all

        Returns:
            dict with curve points and metadata
        """
        # Read AQI and fan_speed data
        aqi_data = self.log_manager.read_logs_for_analysis("aqi", hours)
        fan_data = self.log_manager.read_logs_for_analysis("fan_speed", hours)

        if len(aqi_data) < 10 or len(fan_data) < 10:
            return {"error": "Insufficient data", "aqi_points": len(aqi_data), "fan_points": len(fan_data)}

        # Filter by weekdays if specified
        if weekdays is not None:
            aqi_data = [d for d in aqi_data if d["timestamp"].weekday() in weekdays]
            fan_data = [d for d in fan_data if d["timestamp"].weekday() in weekdays]

        if len(aqi_data) < 10 or len(fan_data) < 10:
            return {"error": "Insufficient data for selected weekdays"}

        # Align data by timestamp (within 30 second tolerance)
        aligned_data = []
        fan_index = 0

        for aqi_point in aqi_data:
            while fan_index < len(fan_data) - 1:
                time_diff = abs((fan_data[fan_index]["timestamp"] - aqi_point["timestamp"]).total_seconds())
                next_diff = abs((fan_data[fan_index + 1]["timestamp"] - aqi_point["timestamp"]).total_seconds())

                if next_diff < time_diff:
                    fan_index += 1
                else:
                    break

            if fan_index < len(fan_data):
                time_diff = abs((fan_data[fan_index]["timestamp"] - aqi_point["timestamp"]).total_seconds())
                if time_diff <= 30:
                    aligned_data.append({
                        "aqi": aqi_point["value"],
                        "fan_speed": fan_data[fan_index]["value"],
                        "timestamp": aqi_point["timestamp"]
                    })

        if len(aligned_data) < 10:
            return {"error": "Could not align enough data points"}

        # Optionally factor in outside AQI
        if include_outside_aqi:
            outside_data = self.log_manager.read_logs_for_analysis("outside_aqi", hours)
            # This would add complexity weighting - simplified for now

        # Group by AQI buckets and compute average fan speed
        buckets = {}
        aqi_thresholds = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600]

        for point in aligned_data:
            aqi = point["aqi"]
            fan = point["fan_speed"]

            # Find bucket
            bucket = 0
            for threshold in aqi_thresholds:
                if aqi >= threshold:
                    bucket = threshold
                else:
                    break

            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(fan)

        # Calculate curve points with cutoff
        curve_points = []
        for aqi_value in sorted(buckets.keys()):
            speeds = sorted(buckets[aqi_value])
            if speeds:
                # Use percentile based on cutoff
                cutoff_index = int(len(speeds) * max_cutoff) - 1
                cutoff_index = max(0, min(cutoff_index, len(speeds) - 1))
                fan_speed = speeds[cutoff_index]
                curve_points.append([aqi_value, round(fan_speed, 1)])

        if len(curve_points) < 2:
            return {"error": "Could not generate enough curve points"}

        # Ensure curve is monotonically increasing (fan speed should increase with AQI)
        for i in range(1, len(curve_points)):
            if curve_points[i][1] < curve_points[i-1][1]:
                curve_points[i][1] = curve_points[i-1][1]

        return {
            "success": True,
            "curve": curve_points,
            "data_points": len(aligned_data),
            "hours_analyzed": hours,
            "buckets_used": len(buckets),
            "include_outside_aqi": include_outside_aqi,
            "max_cutoff": max_cutoff,
            "weekdays": weekdays
        }


# Initialize log manager
log_manager = LogManager(LOGS_DIR, MAX_ROWS_PER_FILE)
analysis_engine = AnalysisEngine(log_manager)


def interpolate_curve(aqi: float, curve: list) -> int:
    """Interpolate fan speed percentage from AQI using custom curve"""
    if not curve or len(curve) < 2:
        return 30  # default
    
    # Sort curve by AQI
    sorted_curve = sorted(curve, key=lambda x: x[0])
    
    # If AQI is below first point, return first point's percent
    if aqi <= sorted_curve[0][0]:
        return int(sorted_curve[0][1])
    
    # If AQI is above last point, return last point's percent
    if aqi >= sorted_curve[-1][0]:
        return int(sorted_curve[-1][1])
    
    # Find the two points to interpolate between
    for i in range(len(sorted_curve) - 1):
        aqi1, percent1 = sorted_curve[i]
        aqi2, percent2 = sorted_curve[i + 1]
        
        if aqi1 <= aqi <= aqi2:
            # Linear interpolation
            if aqi2 == aqi1:
                return int(percent1)
            ratio = (aqi - aqi1) / (aqi2 - aqi1)
            percent = percent1 + ratio * (percent2 - percent1)
            return int(max(0, min(100, percent)))
    
    return 30


def compute_auto_percent(aqi: float, max_percent: int, curve_type: str) -> int:
    """Compute auto mode percentage based on AQI and curve type"""
    if aqi <= 0:
        return 0
    
    # Normalize AQI to 0-1 range (max AQI is 600)
    normalized = min(1.0, aqi / 600.0)
    
    if curve_type == "linear":
        percent = normalized * max_percent
    elif curve_type == "exponential":
        # Exponential: more aggressive at higher AQI
        percent = (normalized ** 2) * max_percent
    elif curve_type == "logarithmic":
        # Logarithmic: more gradual increase
        import math
        percent = (math.log(1 + normalized * 9) / math.log(10)) * max_percent
    elif curve_type == "quadratic":
        # Quadratic: smooth curve
        percent = (normalized ** 1.5) * max_percent
    else:
        # Default to linear
        percent = normalized * max_percent
    
    return int(max(0, min(100, percent)))


def compute_target_percent(aqi: float | None, mode: str, auto_max: int, auto_curve: str, 
                          constant_percent: int, curve: list) -> int:
    """Compute target fan speed percentage based on mode"""
    if aqi is None:
        return 0
    
    if mode == "auto":
        return compute_auto_percent(aqi, auto_max, auto_curve)
    elif mode == "curve":
        return interpolate_curve(aqi, curve)
    elif mode == "constant":
        return constant_percent
    elif mode == "sleep":
        # Lowest possible - use level 1 (minimum)
        return int((1 / 14.0) * 100)  # ~7%
    else:
        return 0


def get_weather(location: str) -> dict:
    """Get weather data from Open-Meteo API (free, no API key needed)"""
    # Coordinates for locations
    coords = {
        "belgrade": {"lat": 44.7866, "lon": 20.4489},
    }
    
    if location.lower() not in coords:
        return {"error": "Unknown location"}
    
    coord = coords[location.lower()]
    
    # Check cache
    with weather_lock:
        if location in weather_cache:
            cache_time = weather_cache_time.get(location, 0)
            if time.time() - cache_time < WEATHER_CACHE_DURATION:
                return weather_cache[location]
    
    try:
        # Current weather
        current_url = f"https://api.open-meteo.com/v1/forecast?latitude={coord['lat']}&longitude={coord['lon']}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Europe/Belgrade"
        current_resp = requests.get(current_url, timeout=5)
        current_data = current_resp.json() if current_resp.status_code == 200 else {}
        
        # Forecast (4 days)
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={coord['lat']}&longitude={coord['lon']}&daily=temperature_2m_max,temperature_2m_min,weather_code&timezone=Europe/Belgrade&forecast_days=5"
        forecast_resp = requests.get(forecast_url, timeout=5)
        forecast_data = forecast_resp.json() if forecast_resp.status_code == 200 else {}
        
        # Weather code to condition mapping
        weather_conditions = {
            0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing Rime Fog",
            51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
            56: "Light Freezing Drizzle", 57: "Dense Freezing Drizzle",
            61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            66: "Light Freezing Rain", 67: "Heavy Freezing Rain",
            71: "Slight Snow", 73: "Moderate Snow", 75: "Heavy Snow",
            77: "Snow Grains", 80: "Slight Rain Showers", 81: "Moderate Rain Showers",
            82: "Violent Rain Showers", 85: "Slight Snow Showers", 86: "Heavy Snow Showers",
            95: "Thunderstorm", 96: "Thunderstorm with Hail", 99: "Thunderstorm with Heavy Hail"
        }
        
        current = current_data.get("current", {})
        daily = forecast_data.get("daily", {})
        
        result = {
            "location": location,
            "current": {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "condition": weather_conditions.get(current.get("weather_code", 0), "Unknown"),
                "weather_code": current.get("weather_code", 0)
            },
            "forecast": []
        }
        
        # Process forecast (skip today, get next 4 days)
        if daily.get("time") and len(daily["time"]) > 1:
            for i in range(1, min(5, len(daily["time"]))):
                result["forecast"].append({
                    "date": daily["time"][i],
                    "temp_max": daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
                    "temp_min": daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
                    "condition": weather_conditions.get(daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else 0, "Unknown"),
                    "weather_code": daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else 0
                })
        
        # Cache result
        with weather_lock:
            weather_cache[location] = result
            weather_cache_time[location] = time.time()
        
        return result
    except Exception as e:
        print(f"Weather API error for {location}: {e}")
        return {"error": str(e)}

def get_outside_aqi() -> dict:
    """Get outside AQI from Open-Meteo API using configured location"""
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={AIR_QUALITY_LATITUDE}&longitude={AIR_QUALITY_LONGITUDE}&current=pm10,pm2_5&timezone={AIR_QUALITY_TIMEZONE}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            pm25 = current.get("pm2_5")
            pm10 = current.get("pm10")

            aqi = None
            if pm25:
                # US AQI calculation for PM2.5
                if pm25 <= 12:
                    aqi = int((pm25 / 12) * 50)
                elif pm25 <= 35.4:
                    aqi = int(50 + ((pm25 - 12) / (35.4 - 12)) * 50)
                elif pm25 <= 55.4:
                    aqi = int(100 + ((pm25 - 35.4) / (55.4 - 35.4)) * 50)
                elif pm25 <= 150.4:
                    aqi = int(150 + ((pm25 - 55.4) / (150.4 - 55.4)) * 100)
                elif pm25 <= 250.4:
                    aqi = int(200 + ((pm25 - 150.4) / (250.4 - 150.4)) * 100)
                else:
                    aqi = int(300 + ((pm25 - 250.4) / (350.4 - 250.4)) * 100)

            return {
                "aqi": aqi,
                "pm25": pm25,
                "pm10": pm10,
                "location": AIR_QUALITY_CITY,
                "timestamp": current.get("time", "")
            }
    except Exception as e:
        print(f"Outside AQI API error: {e}")

    return {"error": "Unable to fetch AQI data"}

def get_outside_air_quality_history() -> list:
    """Get historical air quality data for configured location (last 24 hours)"""
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={AIR_QUALITY_LATITUDE}&longitude={AIR_QUALITY_LONGITUDE}&hourly=pm10,pm2_5,temperature_2m&timezone={AIR_QUALITY_TIMEZONE}&forecast_days=1&past_days=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            pm25 = hourly.get("pm2_5", [])
            pm10 = hourly.get("pm10", [])
            temp = hourly.get("temperature_2m", [])

            result = []
            now = dt.datetime.now()
            for i in range(len(times)):
                time_str = times[i]
                try:
                    time_obj = dt.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    # Get last 24 hours of data
                    if (now - time_obj.replace(tzinfo=None)).total_seconds() <= 86400:  # 24 hours
                        pm25_val = pm25[i] if i < len(pm25) else None
                        pm10_val = pm10[i] if i < len(pm10) else None
                        temp_val = temp[i] if i < len(temp) else None

                        aqi = None
                        if pm25_val:
                            if pm25_val <= 12:
                                aqi = int((pm25_val / 12) * 50)
                            elif pm25_val <= 35.4:
                                aqi = int(50 + ((pm25_val - 12) / (35.4 - 12)) * 50)
                            elif pm25_val <= 55.4:
                                aqi = int(100 + ((pm25_val - 35.4) / (55.4 - 35.4)) * 50)
                            elif pm25_val <= 150.4:
                                aqi = int(150 + ((pm25_val - 55.4) / (150.4 - 55.4)) * 100)
                            elif pm25_val <= 250.4:
                                aqi = int(200 + ((pm25_val - 150.4) / (250.4 - 150.4)) * 100)
                            else:
                                aqi = int(300 + ((pm25_val - 250.4) / (350.4 - 250.4)) * 100)

                        result.append({
                            "timestamp": time_str,
                            "aqi": aqi,
                            "pm25": pm25_val,
                            "pm10": pm10_val,
                            "temperature": temp_val
                        })
                except:
                    continue

            return sorted(result, key=lambda x: x["timestamp"])
    except Exception as e:
        print(f"Outside air quality history error: {e}")

    return []

def check_sleep_schedule() -> tuple[str | None, bool]:
    """Check if we should be in sleep mode based on schedule
    Returns: (mode_override, is_sleep_time)
    """
    if not sleep_schedule.get("enabled", False):
        return None, False
    
    now = dt.datetime.now()
    current_hour = now.hour
    start_hour = sleep_schedule.get("start_hour", 22)
    end_hour = sleep_schedule.get("end_hour", 7)
    
    is_sleep_time = False
    
    # Handle overnight schedule (e.g., 22:00 to 7:00)
    if start_hour > end_hour:
        # Overnight schedule
        if current_hour >= start_hour or current_hour < end_hour:
            is_sleep_time = True
    else:
        # Same-day schedule
        if start_hour <= current_hour < end_hour:
            is_sleep_time = True
    
    if is_sleep_time:
        return "sleep", True
    else:
        # Not sleep time - return post_sleep_mode if we should switch back
        return sleep_schedule.get("post_sleep_mode", "auto"), False

def control_loop() -> None:
    """Background thread that continuously updates the purifier"""
    while True:
        try:
            status = client.status()
            aqi = status.get("aqi")
            temperature = status.get("temperature")
            humidity = status.get("humidity")
            
            # Store historical data
            with state_lock:
                fan_speed_percent = state.get("current_percent", 0)
            with historical_lock:
                historical_data.append({
                    "timestamp": dt.datetime.now().isoformat(),
                    "aqi": aqi,
                    "temperature": temperature,
                    "humidity": humidity,
                    "fan_speed": fan_speed_percent
                })
            
            with state_lock:
                mode = state["mode"]
                auto_max = state.get("auto_max_percent", 100)
                auto_curve = state.get("auto_curve_type", "linear")
                constant_percent = state.get("constant_percent", 50)
                # Use selected curve if in curve mode
                if mode == "curve":
                    selected_curve_name = state.get("selected_curve_name", "Silent")
                    with curves_lock:
                        if selected_curve_name in saved_curves:
                            curve = saved_curves[selected_curve_name]["points"]
                        else:
                            curve = state.get("current_curve", [])
                else:
                    curve = state.get("current_curve", [])
                mode_change_in_progress = state.get("mode_change_in_progress", False)
            
            # Check sleep schedule
            schedule_override, is_sleep_time = check_sleep_schedule()
            if is_sleep_time:
                # Override mode to sleep if schedule is active
                effective_mode = "sleep"
            elif schedule_override and schedule_override != mode and mode == "sleep" and sleep_schedule.get("enabled", False):
                # Sleep time ended, switch to post_sleep_mode if we're still in sleep
                # Only switch if we're currently in sleep mode and schedule is enabled (to avoid overriding manual changes)
                with state_lock:
                    if state["mode"] == "sleep":
                        state["mode"] = schedule_override
                        save_state()
                effective_mode = schedule_override
            else:
                effective_mode = mode
            
            # CRITICAL: Skip control loop entirely if user is changing mode
            if mode_change_in_progress:
                time.sleep(POLL_INTERVAL)
                continue
            
            target_percent = compute_target_percent(aqi, effective_mode, auto_max, auto_curve, constant_percent, curve)

            # Temperature Assist: optionally reduce fan speed when it's cold
            try:
                with state_lock:
                    temp_assist_enabled = bool(state.get("temp_assist_enabled", False))
                    temp_assist_threshold_c = state.get("temp_assist_threshold_c", 18.0)
                    temp_assist_reduce_percent = state.get("temp_assist_reduce_percent", 10)

                if temp_assist_enabled and temperature is not None:
                    if float(temperature) < float(temp_assist_threshold_c):
                        target_percent = max(0, min(100, int(round(target_percent - float(temp_assist_reduce_percent)))))
            except Exception:
                pass
            
            # All modes use favorite mode on device, just control fan speed
            # IMPORTANT: Always use apply_percent which sets device to favorite mode
            if effective_mode in ["auto", "curve", "constant", "sleep"]:
                # CRITICAL: Double-check mode hasn't changed before applying
                with state_lock:
                    current_mode_check = state["mode"]
                    last_mode_change_check = state.get("last_mode_change")
                
                # Re-check sleep schedule
                schedule_override_check, is_sleep_time_check = check_sleep_schedule()
                if is_sleep_time_check:
                    effective_mode_check = "sleep"
                else:
                    effective_mode_check = current_mode_check
                
                # Only apply if mode is still valid and no recent change
                should_apply = (effective_mode_check in ["auto", "curve", "constant", "sleep"])
                if last_mode_change_check:
                    try:
                        change_time = dt.datetime.fromisoformat(last_mode_change_check)
                        time_since_change = (dt.datetime.now() - change_time).total_seconds()
                        if time_since_change < 2:
                            with state_lock:
                                final_mode_check = state["mode"]
                            should_apply = (final_mode_check in ["auto", "curve", "constant", "sleep"])
                    except (ValueError, TypeError):
                        pass
                
                if should_apply:
                    # Apply the target percent (this sets device to favorite mode internally)
                    applied_level = client.apply_percent(target_percent)
                    # Update current percent for display
                    with state_lock:
                        state["current_percent"] = target_percent
                else:
                    applied_level = None
            else:
                # For auto mode, only set the mode if device is not already in auto
                # Don't call set_mode every loop iteration to avoid conflicts
                # Only check/set auto mode occasionally (every 20th iteration = 100 seconds) to reduce conflicts
                # Unknown mode - don't do anything
                applied_level = None
            
            with state_lock:
                state["last_aqi"] = aqi
                state["last_temperature"] = temperature
                state["last_humidity"] = humidity
                state["last_target_percent"] = target_percent
                state["last_applied_level"] = applied_level
                state["last_status"] = status
                state["last_update"] = dt.datetime.now().isoformat(timespec="seconds")
                state["last_error"] = None
                # Update current percent from device status if available
                if applied_level is not None:
                    state["current_percent"] = int((applied_level / LEVEL_MAX) * 100)
        except Exception as exc:
            with state_lock:
                state["last_error"] = str(exc)
        time.sleep(POLL_INTERVAL)


def logging_loop() -> None:
    """Background thread that logs data to CSV files"""
    last_indoor_log = 0
    last_outdoor_log = 0
    outdoor_interval = 300  # 5 minutes for outdoor data

    while True:
        try:
            with state_lock:
                logging_enabled = state.get("logging_enabled", False)
                log_interval = state.get("log_interval", MIN_LOG_INTERVAL)

            if logging_enabled:
                current_time = time.time()

                # Log indoor values at configured interval
                if current_time - last_indoor_log >= log_interval:
                    with state_lock:
                        aqi = state.get("last_aqi")
                        temperature = state.get("last_temperature")
                        humidity = state.get("last_humidity")
                        fan_speed = state.get("current_percent")

                    log_manager.log_value("aqi", aqi)
                    log_manager.log_value("temperature", temperature)
                    log_manager.log_value("humidity", humidity)
                    log_manager.log_value("fan_speed", fan_speed)

                    last_indoor_log = current_time

                # Log outdoor values every 5 minutes
                if current_time - last_outdoor_log >= outdoor_interval:
                    try:
                        outside_aqi_data = get_outside_aqi()
                        if "aqi" in outside_aqi_data:
                            log_manager.log_value("outside_aqi", outside_aqi_data["aqi"])

                        # Get outside temp from weather data
                        weather_data = get_weather("belgrade")
                        if "current" in weather_data and "temperature" in weather_data["current"]:
                            log_manager.log_value("outside_temp", weather_data["current"]["temperature"])
                    except Exception as e:
                        logger.warning(f"Error logging outdoor data: {e}")

                    last_outdoor_log = current_time

        except Exception as exc:
            logger.error(f"Error in logging loop: {exc}")

        time.sleep(1)  # Check every second for responsiveness


app = Flask(__name__)


# HTML Template with interactive curve editor
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Air Purifier Control</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #000000;
            min-height: 100vh;
            padding: 0;
            color: #ffffff;
            overflow-x: hidden;
            overflow-y: auto;
        }

        .app-container {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            width: 100%;
        }
        
        /* Top Status Bar */
        .top-status {
            background: #0a0a0a;
            border-bottom: 1px solid #1a1a1a;
            padding: 15px 20px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            z-index: 100;
        }
        
        .status-row {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .status-row:last-child {
            margin-bottom: 0;
        }
        
        .status-box {
            text-align: center;
            flex: 1;
        }
        
        .status-label {
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 3px;
        }
        
        .status-value {
            font-size: 28px;
            font-weight: 700;
            line-height: 1;
        }
        
        .power-display {
            font-size: 40px;
            font-weight: 700;
        }
        
        /* Weather Section */
        .weather-section {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 12px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .weather-location {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            opacity: 0.9;
        }
        
        .weather-current {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .weather-temp {
            font-size: 24px;
            font-weight: 700;
        }
        
        .weather-condition {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .weather-forecast {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding: 5px 0;
        }
        
        .forecast-day {
            min-width: 60px;
            text-align: center;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            padding: 8px 5px;
        }
        
        .forecast-date {
            font-size: 10px;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .forecast-temp {
            font-size: 14px;
            font-weight: 600;
        }
        
        .forecast-min {
            font-size: 11px;
            opacity: 0.7;
        }
        
        /* Historical Graph */
        .graph-section {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 12px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .graph-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .graph-container {
            position: relative;
            height: 200px;
            background: #000000;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        .graph-container:last-child {
            margin-bottom: 0;
        }
        
        /* Schedule Section */
        .schedule-section {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 12px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .schedule-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .schedule-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .toggle-slider {
            background-color: #dfe7e2;
        }

        .toggle-slider:before {
            background-color: #ffffff;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        }

        input:checked + .toggle-slider {
            background-color: var(--accent);
        }
        
        .schedule-time-inputs {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .time-input {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            color: #fff;
            padding: 8px;
            font-size: 14px;
            width: 60px;
            text-align: center;
        }
        
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #1a1a1a;
            transition: .4s;
            border-radius: 24px;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .toggle-slider {
            background-color: #667eea;
        }
        
        input:checked + .toggle-slider:before {
            transform: translateX(26px);
        }
        
        /* Collapsible Sections */
        .collapsible-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            padding: 12px 15px;
            background: #0a0a0a;
            border-bottom: 1px solid #1a1a1a;
            user-select: none;
            transition: background 0.2s;
        }
        
        .collapsible-header:hover {
            background: #1a1a1a;
        }
        
        .collapsible-title {
            font-size: 14px;
            font-weight: 600;
            opacity: 0.9;
        }
        
        .collapse-icon {
            font-size: 18px;
            transition: transform 0.3s;
            opacity: 0.7;
        }
        
        .collapsible-header.collapsed .collapse-icon {
            transform: rotate(-90deg);
        }
        
        .collapsible-content {
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.3s ease-out, padding 0.3s ease-out;
            padding: 15px;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
            padding: 0 15px;
        }
        
        /* Main Layout */
        .main-layout {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        /* Side Menu */
        .side-menu {
            width: 80px;
            background: #0a0a0a;
            border-right: 1px solid #1a1a1a;
            display: flex;
            flex-direction: column;
            padding: 10px 0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.5);
            z-index: 50;
        }
        
        .menu-item {
            width: 100%;
            padding: 20px 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 3px solid transparent;
            touch-action: manipulation;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .menu-item:active {
            transform: scale(0.95);
        }
        
        .menu-item.active {
            background: rgba(102, 126, 234, 0.15);
            border-left-color: #667eea;
        }
        
        .menu-icon {
            font-size: 28px;
            margin-bottom: 4px;
        }
        
        .menu-label {
            font-size: 11px;
            opacity: 0.8;
        }
        
        .menu-item.active .menu-label {
            opacity: 1;
            font-weight: 600;
        }
        
        /* Content Area - Settings Panel */
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: #000000;
        }
        
        .settings-panel {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        .settings-header {
            background: #0a0a0a;
            border-bottom: 1px solid #1a1a1a;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        
        .settings-title {
            font-size: 24px;
            font-weight: 700;
        }
        
        .close-settings {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            color: #fff;
            width: 36px;
            height: 36px;
            border-radius: 18px;
            font-size: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .close-settings:hover {
            background: #2a2a2a;
        }
        
        .settings-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #000000;
        }
        
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            display: block;
            font-size: 14px;
            margin-bottom: 8px;
            opacity: 0.8;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .slider {
            flex: 1;
            height: 8px;
            border-radius: 4px;
            background: #1a1a1a;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .slider::-moz-range-thumb {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .number-input {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            color: #fff;
            padding: 12px;
            font-size: 16px;
            width: 80px;
            text-align: center;
        }
        
        .select-input {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            color: #fff;
            padding: 12px;
            font-size: 16px;
            width: 100%;
        }
        
        .btn {
            background: #667eea;
            border: none;
            color: #fff;
            padding: 14px 24px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            touch-action: manipulation;
            transition: background 0.2s;
        }
        
        .btn:hover {
            background: #5568d3;
        }
        
        .btn:active {
            transform: scale(0.98);
        }
        
        .btn-secondary {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
        }
        
        .btn-secondary:hover {
            background: #2a2a2a;
        }
        
        /* Curve Editor */
        .chart-container {
            position: relative;
            height: 250px;
            margin: 20px 0;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 12px;
            padding: 10px;
        }
        
        .curve-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .point-list {
            max-height: 150px;
            overflow-y: auto;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
        
        .point-item {
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 8px;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            margin-bottom: 6px;
        }
        
        .point-input {
            flex: 1;
            background: #000000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #fff;
            padding: 8px;
            font-size: 14px;
        }
        
        .delete-point {
            background: #ff4757;
            border: none;
            color: #fff;
            width: 28px;
            height: 28px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 18px;
        }
        
        .saved-curves {
            margin-top: 20px;
        }
        
        .curve-item {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .curve-item-name {
            font-weight: 600;
            font-size: 14px;
        }
        
        .curve-item-actions {
            display: flex;
            gap: 6px;
        }
        
        .btn-small {
            padding: 6px 12px;
            font-size: 12px;
            width: auto;
            margin: 0;
        }
        
        /* Toast Messages */
        .toast {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            color: #fff;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 2000;
            display: none;
            animation: slideInUp 0.3s;
        }
        
        .toast.show {
            display: block;
        }
        
        .toast.error {
            background: #ff4757;
        }
        
        .toast.success {
            background: #2ed573;
        }
        
        @keyframes slideInUp {
            from {
                transform: translateY(100px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        /* ============= Modal Overlay ============= */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            z-index: 3000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .modal-overlay.show {
            display: flex;
        }

        .modal-container {
            background: var(--card);
            border-radius: 16px;
            max-width: 700px;
            width: 100%;
            max-height: 80vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
        }

        .modal-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--ink);
        }

        .modal-close {
            background: var(--card-alt);
            border: 1px solid var(--border);
            color: var(--ink);
            width: 32px;
            height: 32px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .modal-body {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }

        /* ============= Log Explorer Styles ============= */
        .log-folder {
            margin-bottom: 12px;
        }

        .log-folder-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            background: var(--card-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            color: var(--ink);
        }

        .log-folder-header:hover {
            background: var(--accent-soft);
        }

        .log-folder-icon {
            font-size: 16px;
        }

        .log-folder-name {
            flex: 1;
        }

        .log-folder-count {
            font-size: 12px;
            opacity: 0.7;
        }

        .log-folder-files {
            margin-left: 20px;
            margin-top: 8px;
            display: none;
        }

        .log-folder.expanded .log-folder-files {
            display: block;
        }

        .log-file {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 6px;
        }

        .log-file-icon {
            opacity: 0.6;
        }

        .log-file-info {
            flex: 1;
            min-width: 0;
        }

        .log-file-name {
            font-size: 13px;
            font-weight: 500;
            color: var(--ink);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .log-file-meta {
            font-size: 11px;
            color: var(--muted);
        }

        .log-file-actions {
            display: flex;
            gap: 6px;
        }

        .log-file-btn {
            padding: 4px 10px;
            font-size: 11px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--card-alt);
            color: var(--ink);
            cursor: pointer;
        }

        .log-file-btn:hover {
            background: var(--accent-soft);
        }

        .log-preview {
            background: var(--card-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-top: 12px;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            max-height: 300px;
            overflow: auto;
            white-space: pre;
            color: var(--ink);
        }

        /* ============= Stylish Checkbox ============= */
        .checkbox-stylish {
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            user-select: none;
        }

        .checkbox-stylish input {
            display: none;
        }

        .checkbox-stylish .checkmark {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            background: var(--card);
        }

        .checkbox-stylish input:checked + .checkmark {
            background: var(--accent);
            border-color: var(--accent);
        }

        .checkbox-stylish .checkmark::after {
            content: "";
            display: none;
            width: 5px;
            height: 10px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
            margin-bottom: 2px;
        }

        .checkbox-stylish input:checked + .checkmark::after {
            display: block;
        }

        .checkbox-stylish .checkbox-label {
            font-size: 14px;
            color: var(--ink);
        }

        /* ============= Analysis Section ============= */
        .analysis-status {
            padding: 12px;
            background: var(--card-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .analysis-status-item {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 13px;
        }

        .analysis-status-label {
            color: var(--muted);
        }

        .analysis-status-value {
            color: var(--ink);
            font-weight: 500;
        }

        .weekday-selector {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin: 10px 0;
        }

        .weekday-btn {
            padding: 6px 12px;
            font-size: 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--card);
            color: var(--ink);
            cursor: pointer;
            transition: all 0.2s;
        }

        .weekday-btn.active {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }

        .curve-preview-container {
            margin-top: 15px;
            padding: 15px;
            background: var(--card-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        .curve-preview-chart {
            height: 200px;
            margin-bottom: 10px;
        }

        .curve-preview-info {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 10px;
        }

        /* ============= Logging Section Styles ============= */
        .logging-controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .logging-toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logging-toggle-label {
            font-size: 14px;
            color: var(--ink);
        }

        .logging-interval-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logging-interval-label {
            font-size: 13px;
            color: var(--muted);
            min-width: 80px;
        }

        /* ============= Temperature Assist Styles ============= */
        .temp-assist-controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .temp-assist-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .temp-assist-row-label {
            font-size: 13px;
            color: var(--muted);
            min-width: 110px;
        }

        .temp-assist-row-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1 1 auto;
            justify-content: flex-end;
        }

        /* --- Visual Refresh Overrides --- */
        :root {
            color-scheme: light;
            --bg: #f5f1ea;
            --ink: #1d2b24;
            --muted: rgba(29, 43, 36, 0.55);
            --accent: #2c9c8f;
            --accent-strong: #1c6c62;
            --accent-soft: #eaf6f3;
            --card: #ffffff;
            --card-alt: #f7faf8;
            --card-glass: rgba(255, 255, 255, 0.86);
            --border: rgba(29, 43, 36, 0.12);
            --shadow: 0 20px 55px rgba(29, 43, 36, 0.12);
            --chart-text: #1d2b24;
            --chart-grid: rgba(29, 43, 36, 0.12);
            --chart-aqi: #d66c53;
            --chart-temp: #2c9c8f;
            --chart-humidity: #3b78b9;
            --chart-fan: #e0b457;
        }

        html[data-theme="dark"] {
            color-scheme: dark;
            --bg: #0f1311;
            --ink: #f5f4f0;
            --muted: rgba(245, 244, 240, 0.55);
            --accent: #3bb8a5;
            --accent-strong: #2a8f82;
            --accent-soft: rgba(59, 184, 165, 0.18);
            --card: #151a18;
            --card-alt: #1b221f;
            --card-glass: rgba(18, 22, 20, 0.88);
            --border: rgba(245, 244, 240, 0.12);
            --shadow: 0 26px 60px rgba(0, 0, 0, 0.45);
            --chart-text: #f5f4f0;
            --chart-grid: rgba(245, 244, 240, 0.14);
            --chart-aqi: #f08e74;
            --chart-temp: #53d1bc;
            --chart-humidity: #6aa8e6;
            --chart-fan: #f2c86d;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg);
            color: var(--ink);
            position: relative;
            height: auto;
            width: auto;
            overflow-y: auto;
            transition: background 0.3s ease, color 0.3s ease;
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background:
                radial-gradient(circle at 10% 0%, rgba(44, 156, 143, 0.18), transparent 55%),
                radial-gradient(circle at 90% 15%, rgba(224, 180, 87, 0.22), transparent 45%),
                radial-gradient(circle at 30% 90%, rgba(214, 108, 83, 0.18), transparent 60%);
            pointer-events: none;
            z-index: -2;
        }

        body::after {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(29, 43, 36, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(29, 43, 36, 0.04) 1px, transparent 1px);
            background-size: 44px 44px;
            pointer-events: none;
            z-index: -1;
            opacity: 0.35;
        }

        html[data-theme="dark"] body::before {
            background:
                radial-gradient(circle at 10% 0%, rgba(59, 184, 165, 0.2), transparent 55%),
                radial-gradient(circle at 90% 15%, rgba(242, 200, 109, 0.15), transparent 50%),
                radial-gradient(circle at 30% 90%, rgba(240, 142, 116, 0.18), transparent 60%);
        }

        html[data-theme="dark"] body::after {
            background-image:
                linear-gradient(rgba(245, 244, 240, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(245, 244, 240, 0.04) 1px, transparent 1px);
        }

        .app-container {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            width: 100%;
        }

        .dashboard-masonry {
            max-width: 2200px;
            margin: 0 auto;
            padding: 18px 20px 40px;
            column-gap: 18px;
            column-width: 520px;
            column-fill: balance;
        }

        .dashboard-masonry > * {
            display: inline-block;
            width: 100%;
            vertical-align: top;
            margin: 0 0 18px;
            break-inside: avoid;
            page-break-inside: avoid;
            -webkit-column-break-inside: avoid;
        }

        .top-status {
            background: var(--card-glass);
            border: 1px solid var(--border);
            border-radius: 0 0 18px 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 10;
            backdrop-filter: blur(10px);
            width: 100vw;
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
        }

        .status-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .mode-pill {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-strong);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .mode-pill[data-mode="curve"] { background: #e9eefb; color: #3f5fb7; }
        .mode-pill[data-mode="constant"] { background: #fff3d7; color: #9b6a1f; }
        .mode-pill[data-mode="sleep"] { background: #fde9e0; color: #a4513b; }

        .mode-caption {
            font-size: 12px;
            color: var(--muted);
            flex: 1 1 auto;
        }

        .theme-switch {
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .theme-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 600;
        }

        .theme-toggle {
            border: 1px solid var(--border);
            background: var(--card);
            color: var(--ink);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease, border-color 0.2s ease;
        }

        .theme-toggle:hover {
            background: var(--card-alt);
        }

        .status-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }

        .status-box {
            text-align: left;
        }

        .status-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
        }

        .status-value {
            font-family: 'DM Mono', monospace;
            font-size: 36px;
            color: var(--ink);
        }

        .power-display {
            font-family: 'DM Mono', monospace;
            font-size: 42px;
            color: var(--accent-strong);
        }

        .aqi-tag {
            margin-top: 8px;
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: #e8f4f0;
            color: #1c6c62;
        }

        .aqi-tag.moderate { background: #fff2d4; color: #8a5a12; }
        .aqi-tag.caution { background: #ffe6d9; color: #9b4a2a; }
        .aqi-tag.unhealthy { background: #ffd6d6; color: #9a2f2f; }
        .aqi-tag.hazard { background: #f4d6f0; color: #7b2f6f; }

        .main-layout {
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow: visible;
            order: 1;
        }

        .side-menu {
            width: 100%;
            background: transparent;
            border: none;
            box-shadow: none;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 12px;
            padding: 0;
        }

        .menu-item {
            flex: 1 1 160px;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--card);
            padding: 14px;
            align-items: flex-start;
            text-align: left;
            gap: 8px;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .menu-item.active {
            background: var(--accent-soft);
            border-color: var(--accent);
        }

        .menu-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            background: var(--accent-soft);
            color: var(--accent-strong);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 6px;
        }

        .menu-item[data-mode="curve"] .menu-icon { background: #e9eefb; color: #3f5fb7; }
        .menu-item[data-mode="constant"] .menu-icon { background: #fff3d7; color: #9b6a1f; }
        .menu-item[data-mode="sleep"] .menu-icon { background: #fde9e0; color: #a4513b; }

        .menu-label {
            font-size: 14px;
            font-weight: 600;
            opacity: 1;
        }

        .content-area {
            background: transparent;
            overflow: visible;
            flex: none;
        }

        .settings-panel {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--card);
            box-shadow: var(--shadow);
            overflow: hidden;
            height: auto;
            width: 100%;
        }

        .settings-header {
            background: linear-gradient(135deg, rgba(44, 156, 143, 0.08), rgba(255, 255, 255, 0));
            border-bottom: 1px solid var(--border);
        }

        .settings-title {
            font-size: 20px;
            color: var(--ink);
        }

        .settings-content {
            background: transparent;
            overflow: visible;
        }

        .btn {
            background: var(--accent);
        }

        .btn:hover {
            background: var(--accent-strong);
        }

        .btn-secondary {
            background: var(--card-alt);
            border-color: var(--border);
            color: var(--ink);
        }

        .btn-secondary:hover {
            background: var(--card);
        }

        .btn-danger {
            background: #d66c53;
            color: #fff;
            border: none;
        }

        .btn-danger:hover {
            background: #b8493a;
        }

        .chart-container,
        .graph-container {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .number-input,
        .select-input,
        .point-input,
        .time-input {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--ink);
        }

        .slider {
            background: #e5ece8;
        }

        .slider::-webkit-slider-thumb,
        .slider::-moz-range-thumb {
            background: var(--accent);
        }

        .chart-container {
            height: 240px;
        }

        .graph-container {
            height: 170px;
        }

        .point-list,
        .point-item,
        .curve-item {
            background: var(--card-alt);
            border-color: var(--border);
        }

        .weather-section,
        .graph-section,
        .schedule-section {
            background: var(--card-glass);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            margin: 0;
        }

        .weather-forecast {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(66px, 1fr));
            gap: 8px;
        }

        .forecast-day {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 8px 6px;
            text-align: center;
        }

        .forecast-date,
        .forecast-min {
            color: var(--muted);
        }

        .graph-title,
        .schedule-title {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 12px;
        }

        .collapsible-header {
            padding: 14px 16px;
            background: transparent;
        }

        .collapsible-header:hover {
            background: rgba(44, 156, 143, 0.08);
        }

        .collapse-icon {
            font-size: 0;
            width: 16px;
            height: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .collapse-icon::after {
            content: ">";
            font-size: 14px;
            color: var(--muted);
            transition: transform 0.3s ease;
        }

        .collapsible-header.collapsed .collapse-icon::after {
            transform: rotate(-90deg);
        }

        .collapsible-header.collapsed .collapse-icon {
            transform: none;
        }

        .collapsible-content {
            padding: 0 16px 16px;
        }

        .toast {
            background: #1d2b24;
            color: #f5f1ea;
            border-color: rgba(29, 43, 36, 0.25);
        }

        .toast.success { background: #2b7d6f; }
        .toast.error { background: #b8493a; }

        @media (max-width: 980px) {
            .top-status {
                position: static;
            }

            .dashboard-masonry {
                column-count: 1;
                column-width: auto;
                padding: 16px 16px 32px;
            }

            .side-menu {
                flex-wrap: wrap;
            }
        }

        @media (max-width: 720px) {
            .status-row {
                grid-template-columns: 1fr;
            }

            .status-meta {
                align-items: flex-start;
            }

            .mode-caption {
                flex: 1 1 100%;
            }

            .theme-switch {
                width: 100%;
                justify-content: space-between;
            }

            .theme-label {
                display: none;
            }

            .side-menu {
                flex-wrap: nowrap;
                overflow-x: auto;
                scroll-snap-type: x mandatory;
                padding-bottom: 6px;
            }

            .menu-item {
                min-width: 160px;
                flex: 0 0 auto;
                scroll-snap-align: start;
            }

            .status-value {
                font-size: 30px;
            }

            .power-display {
                font-size: 34px;
            }

            .settings-header {
                padding: 16px;
            }

            .settings-content {
                padding: 16px;
            }

            .side-nav {
                gap: 8px;
            }

            .nav-brand {
                display: none;
            }

            .nav-item {
                flex: 1 1 auto;
                justify-content: center;
            }
        }

        @media (max-width: 640px) {
            .app-container {
                padding: 16px 16px 32px;
            }

            .chart-container {
                height: 210px;
            }

            .graph-container {
                height: 150px;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Top Status Bar -->
        <div class="top-status">
            <div class="status-meta">
                <div class="mode-pill" id="mode-pill">Mode: --</div>
                <div class="mode-caption">Live room readings</div>
                <div class="theme-switch">
                    <span class="theme-label">Theme</span>
                    <button class="theme-toggle" id="theme-toggle" type="button" onclick="toggleTheme()">Light</button>
                </div>
            </div>
            <div class="status-row">
                <div class="status-box">
                    <div class="status-label">AQI</div>
                    <div class="status-value" id="aqi">--</div>
                    <div class="aqi-tag" id="aqi-label">--</div>
                </div>
                <div class="status-box">
                    <div class="status-label">Power Level</div>
                    <div class="power-display" id="power-level">0%</div>
                </div>
            </div>
            <div class="status-row">
                <div class="status-box">
                    <div class="status-label">Temperature</div>
                    <div class="status-value" id="temperature">--</div>
                </div>
                <div class="status-box">
                    <div class="status-label">Humidity</div>
                    <div class="status-value" id="humidity">--</div>
                </div>
            </div>
        </div>

        <div class="dashboard-masonry">
        <!-- Main Layout -->
        <div class="main-layout">
            <!-- Side Menu -->
            <div class="side-menu">
                <div class="menu-item" data-mode="auto" onclick="selectMode('auto')">
                    <div class="menu-icon">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                            <circle cx="12" cy="12" r="8"></circle>
                            <path d="M12 4v2M12 18v2M4 12h2M18 12h2"></path>
                        </svg>
                    </div>
                    <div class="menu-label">Auto</div>
                </div>
                <div class="menu-item" data-mode="curve" onclick="selectMode('curve')">
                    <div class="menu-icon">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                            <path d="M4 16c3-8 6-8 9-4s5 4 7 0"></path>
                            <circle cx="6" cy="16" r="1.5"></circle>
                            <circle cx="13" cy="12" r="1.5"></circle>
                            <circle cx="20" cy="12" r="1.5"></circle>
                        </svg>
                    </div>
                    <div class="menu-label">Curve</div>
                </div>
                <div class="menu-item" data-mode="constant" onclick="selectMode('constant')">
                    <div class="menu-icon">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                            <path d="M5 12h14"></path>
                            <circle cx="12" cy="12" r="6"></circle>
                        </svg>
                    </div>
                    <div class="menu-label">Constant</div>
                </div>
                <div class="menu-item" data-mode="sleep" onclick="selectMode('sleep')">
                    <div class="menu-icon">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                            <path d="M16 14a6 6 0 1 1-8-8 7 7 0 0 0 8 8z"></path>
                        </svg>
                    </div>
                    <div class="menu-label">Sleep</div>
                </div>
            </div>
            
            <!-- Content Area --><!-- Content Area -->
            <div class="content-area">
                <!-- Settings Panel -->
                <div class="settings-panel" id="settings-panel">
                    <div class="settings-header">
                        <div class="settings-title" id="settings-title">Select a Mode</div>
                    </div>
                    <div class="settings-content" id="settings-content">
                        <p style="opacity: 0.6; text-align: center; padding: 40px;">Select a mode above to configure settings</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Temperature Assist -->
        <div class="schedule-section insight-block">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'temp-assist')">
                <div class="collapsible-title">Temperature Assist</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="temp-assist-content-wrapper">
                <div class="temp-assist-controls">
                    <div class="logging-toggle-row">
                        <span class="logging-toggle-label">Enable Temperature Assist</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="temp-assist-enabled" onchange="updateTempAssist()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="temp-assist-row">
                        <span class="temp-assist-row-label">If Temp below:</span>
                        <div class="temp-assist-row-controls">
                            <input type="number" class="number-input" id="temp-assist-threshold" min="-10" max="40" step="0.5" value="18" style="width: 90px;" onchange="updateTempAssist()">
                            <span style="font-size: 12px; color: var(--muted);">&deg;C</span>
                        </div>
                    </div>
                    <div class="temp-assist-row">
                        <span class="temp-assist-row-label">Reduce fan by:</span>
                        <div class="temp-assist-row-controls">
                            <input type="range" class="slider" id="temp-assist-reduce-slider" min="0" max="100" value="10" onchange="syncTempAssistReduce(this.value)">
                            <input type="number" class="number-input" id="temp-assist-reduce-number" min="0" max="100" value="10" style="width: 70px;" onchange="syncTempAssistReduce(this.value)">
                            <span style="font-size: 12px; color: var(--muted);">%</span>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: var(--muted);">Applies after Auto/Curve/Constant/Sleep target speed (subtracts % points).</div>
                </div>
            </div>
        </div>

        <!-- Weather Section -->
        <div class="weather-section insight-block insight-weather" id="weather-section">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'weather')">
                <div class="collapsible-title">Weather - Belgrade</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="weather-content-wrapper">
                <div id="weather-content"></div>
            </div>
        </div>
        
        <!-- Historical Graph -->
        <div class="graph-section insight-block insight-history">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'graph')">
                <div class="collapsible-title">Historical Data - Last 6 Hours</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="graph-content-wrapper">
                <div class="graph-title">AQI</div>
                <div class="graph-container">
                    <canvas id="historicalChartAQI"></canvas>
                </div>
                <div class="graph-title">Temperature</div>
                <div class="graph-container">
                    <canvas id="historicalChartTemp"></canvas>
                </div>
                <div class="graph-title">Humidity</div>
                <div class="graph-container">
                    <canvas id="historicalChartHumidity"></canvas>
                </div>
                <div class="graph-title">Fan Speed (%)</div>
                <div class="graph-container">
                    <canvas id="historicalChartFanSpeed"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Outside Air Quality -->
        <div class="graph-section insight-block insight-outside">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'outside-air')">
                <div class="collapsible-title">Outside Air Quality</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="outside-air-content-wrapper">
                <div style="margin-bottom: 15px; padding: 12px; background: var(--card); border-radius: 12px; border: 1px solid var(--border);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 11px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.08em;">Current AQI</div>
                            <div style="font-size: 24px; font-weight: 700;" id="outside-aqi">--</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.08em;">PM2.5</div>
                            <div style="font-size: 16px;" id="outside-pm25">--</div>
                        </div>
                    </div>
                </div>
                <div class="graph-title">Outside AQI (24h)</div>
                <div class="graph-container">
                    <canvas id="outsideAQIChart"></canvas>
                </div>
                <div class="graph-title">Outside Temperature (24h)</div>
                <div class="graph-container">
                    <canvas id="outsideTempChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Sleep Schedule -->
        <div class="schedule-section insight-block insight-schedule">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'schedule')">
                <div class="collapsible-title">Sleep Mode Schedule</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="schedule-content-wrapper">
                <div class="schedule-controls">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <label style="flex: 1;">Enable Schedule:</label>
                        <label class="toggle-switch">
                            <input type="checkbox" id="schedule-enabled" onchange="updateSchedule()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Sleep Time</label>
                        <div class="schedule-time-inputs">
                            <input type="number" class="time-input" id="schedule-start" min="0" max="23" value="22" onchange="updateSchedule()">
                            <span>to</span>
                            <input type="number" class="time-input" id="schedule-end" min="0" max="23" value="7" onchange="updateSchedule()">
                            <span>hours</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Mode After Sleep</label>
                        <select class="select-input" id="schedule-post-mode" onchange="updateSchedule()">
                            <option value="auto">Auto</option>
                            <option value="curve">Curve</option>
                            <option value="constant">Constant</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data Logging Section -->
        <div class="schedule-section insight-block">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'logging')">
                <div class="collapsible-title">Data Logging</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="logging-content-wrapper">
                <div class="logging-controls">
                    <div class="logging-toggle-row">
                        <span class="logging-toggle-label">Enable Logging</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="logging-enabled" onchange="updateLogging()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="logging-interval-row">
                        <span class="logging-interval-label">Interval:</span>
                        <input type="range" class="slider" id="log-interval-slider" min="10" max="300" value="30" onchange="updateLogInterval(this.value)">
                        <input type="number" class="number-input" id="log-interval-number" min="10" max="300" value="30" style="width: 70px;" onchange="updateLogInterval(this.value)">
                        <span style="font-size: 12px; color: var(--muted);">sec</span>
                    </div>
                    <button class="btn btn-secondary" onclick="openLogExplorer()">
                        Browse Log Files
                    </button>
                </div>
            </div>
        </div>

        <!-- Analysis Section -->
        <div class="schedule-section insight-block">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'analysis')">
                <div class="collapsible-title">Curve Analysis</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="analysis-content-wrapper">
                <div class="analysis-status" id="analysis-data-status">
                    <div class="analysis-status-item">
                        <span class="analysis-status-label">AQI Data:</span>
                        <span class="analysis-status-value" id="analysis-aqi-hours">--</span>
                    </div>
                    <div class="analysis-status-item">
                        <span class="analysis-status-label">Fan Speed Data:</span>
                        <span class="analysis-status-value" id="analysis-fan-hours">--</span>
                    </div>
                </div>

                <div class="form-group">
                    <label class="form-label">Time Period</label>
                    <select class="select-input" id="analysis-period">
                        <option value="48">Last 48 hours</option>
                        <option value="168">Last 7 days</option>
                        <option value="336">Last 14 days</option>
                        <option value="720">Last 30 days</option>
                    </select>
                </div>

                <div class="form-group">
                    <label class="form-label">Filter by Weekdays (optional)</label>
                    <div class="weekday-selector" id="weekday-selector">
                        <button class="weekday-btn" data-day="0" onclick="toggleWeekday(0)">Mon</button>
                        <button class="weekday-btn" data-day="1" onclick="toggleWeekday(1)">Tue</button>
                        <button class="weekday-btn" data-day="2" onclick="toggleWeekday(2)">Wed</button>
                        <button class="weekday-btn" data-day="3" onclick="toggleWeekday(3)">Thu</button>
                        <button class="weekday-btn" data-day="4" onclick="toggleWeekday(4)">Fri</button>
                        <button class="weekday-btn" data-day="5" onclick="toggleWeekday(5)">Sat</button>
                        <button class="weekday-btn" data-day="6" onclick="toggleWeekday(6)">Sun</button>
                    </div>
                </div>

                <label class="checkbox-stylish" style="margin-bottom: 15px;">
                    <input type="checkbox" id="analysis-outside-aqi">
                    <span class="checkmark"></span>
                    <span class="checkbox-label">Include Outside AQI influence</span>
                </label>

                <div class="form-group">
                    <label class="form-label">Max Cutoff: <span id="cutoff-value">100%</span></label>
                    <input type="range" class="slider" id="analysis-cutoff" min="20" max="100" value="100" oninput="document.getElementById('cutoff-value').textContent = this.value + '%'">
                    <div style="font-size: 11px; color: var(--muted); margin-top: 5px;">Lower values create quieter curves by using lower percentile fan speeds</div>
                </div>

                <div class="form-group">
                    <label class="form-label">Curve Name</label>
                    <input type="text" class="select-input" id="analysis-curve-name" placeholder="Generated Curve" value="Generated Curve">
                </div>

                <button class="btn" onclick="generateAnalysisCurve()">
                    Generate Curve
                </button>

                <div class="curve-preview-container" id="curve-preview-container" style="display: none;">
                    <div class="curve-preview-info" id="curve-preview-info"></div>
                    <div class="curve-preview-chart">
                        <canvas id="curve-preview-canvas"></canvas>
                    </div>
                    <button class="btn" onclick="loadGeneratedCurve()">
                        Load in Curve Editor
                    </button>
                </div>
            </div>
        </div>
        </div>
    </div>
    
    <!-- Log Explorer Modal -->
    <div class="modal-overlay" id="log-explorer-modal" onclick="if(event.target === this) closeLogExplorer()">
        <div class="modal-container">
            <div class="modal-header">
                <div class="modal-title">Log Files</div>
                <button class="modal-close" onclick="closeLogExplorer()">&times;</button>
            </div>
            <div class="modal-body" id="log-explorer-body">
                <div style="text-align: center; color: var(--muted);">Loading...</div>
            </div>
        </div>
    </div>

    <!-- Toast Messages -->
    <div id="toast" class="toast"></div>
    
    <script>
        // Global state
        let currentMode = '{{ state.mode }}';
        let chart = null;
        let historicalChartAQI = null;
        let historicalChartTemp = null;
        let historicalChartHumidity = null;
        let historicalChartFanSpeed = null;
        let outsideAQIChart = null;
        let outsideTempChart = null;
        let curvePoints = {{ state.current_curve | tojson }};
        let savedCurvesList = {};

        function formatMode(mode) {
            if (!mode) return '--';
            return mode.charAt(0).toUpperCase() + mode.slice(1);
        }

        function getAqiDescriptor(aqi) {
            if (aqi <= 50) return {label: 'Good', className: 'good'};
            if (aqi <= 100) return {label: 'Moderate', className: 'moderate'};
            if (aqi <= 150) return {label: 'Sensitive', className: 'caution'};
            if (aqi <= 200) return {label: 'Unhealthy', className: 'unhealthy'};
            return {label: 'Hazard', className: 'hazard'};
        }

        function updateAqiLabel(aqi) {
            const label = document.getElementById('aqi-label');
            if (!label) return;
            if (aqi === null || aqi === undefined) {
                label.textContent = '--';
                label.className = 'aqi-tag';
                return;
            }
            const info = getAqiDescriptor(aqi);
            label.textContent = info.label;
            label.className = 'aqi-tag ' + info.className;
        }

        function updateModePill() {
            const pill = document.getElementById('mode-pill');
            if (!pill) return;
            pill.textContent = 'Mode: ' + formatMode(currentMode);
            pill.dataset.mode = currentMode || 'auto';
        }

        function getChartTheme() {
            const styles = getComputedStyle(document.documentElement);
            return {
                text: styles.getPropertyValue('--chart-text').trim() || '#1d2b24',
                grid: styles.getPropertyValue('--chart-grid').trim() || 'rgba(29, 43, 36, 0.12)',
                aqi: styles.getPropertyValue('--chart-aqi').trim() || '#d66c53',
                temp: styles.getPropertyValue('--chart-temp').trim() || '#2c9c8f',
                humidity: styles.getPropertyValue('--chart-humidity').trim() || '#3b78b9',
                fan: styles.getPropertyValue('--chart-fan').trim() || '#e0b457'
            };
        }

        function toRgba(color, alpha) {
            if (!color) return `rgba(0, 0, 0, ${alpha})`;
            if (color.startsWith('#')) {
                const hex = color.replace('#', '');
                const value = hex.length === 3
                    ? hex.split('').map(ch => ch + ch).join('')
                    : hex;
                const intVal = parseInt(value, 16);
                const r = (intVal >> 16) & 255;
                const g = (intVal >> 8) & 255;
                const b = intVal & 255;
                return `rgba(${r}, ${g}, ${b}, ${alpha})`;
            }
            const match = color.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);
            if (match) {
                return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
            }
            return color;
        }

        function setTheme(theme) {
            const nextTheme = theme === 'dark' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', nextTheme);
            localStorage.setItem('theme', nextTheme);

            const toggle = document.getElementById('theme-toggle');
            if (toggle) {
                toggle.textContent = nextTheme === 'dark' ? 'Dark' : 'Light';
                toggle.setAttribute('aria-pressed', nextTheme === 'dark' ? 'true' : 'false');
            }

            refreshCharts();
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'light';
            setTheme(current === 'dark' ? 'light' : 'dark');
        }

        function initTheme() {
            const stored = localStorage.getItem('theme');
            if (stored) {
                setTheme(stored);
                return;
            }
            const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            setTheme(prefersDark ? 'dark' : 'light');
        }

        function refreshCharts() {
            loadHistorical();
            loadOutsideAirQuality();
            updateCurveChartTheme();
        }

        function updateCurveChartTheme() {
            if (!chart) return;
            const theme = getChartTheme();
            chart.data.datasets[0].borderColor = theme.temp;
            chart.data.datasets[0].backgroundColor = toRgba(theme.temp, 0.12);
            chart.data.datasets[0].pointBackgroundColor = theme.aqi;
            chart.data.datasets[0].pointBorderColor = theme.text;
            chart.options.scales.x.ticks.color = theme.text;
            chart.options.scales.y.ticks.color = theme.text;
            chart.options.scales.x.title.color = theme.text;
            chart.options.scales.y.title.color = theme.text;
            chart.options.scales.x.grid.color = theme.grid;
            chart.options.scales.y.grid.color = theme.grid;
            chart.update('none');
        }
        
        // Collapsible functionality
        function toggleCollapse(headerElement, section) {
            const header = headerElement;
            const content = document.getElementById(section + '-content-wrapper');
            if (!content || !header) return;
            
            header.classList.toggle('collapsed');
            content.classList.toggle('collapsed');
        }
        
        // Initialize
        initTheme();
        updateStatus();
        setInterval(updateStatus, 3000);
        loadSavedCurves();
        loadWeather();
        setInterval(loadWeather, 300000); // Update weather every 5 minutes
        loadOutsideAQI();
        setInterval(loadOutsideAQI, 300000); // Update outside AQI every 5 minutes
        setTimeout(() => {
            loadHistorical();
            setInterval(loadHistorical, 60000); // Update historical every minute
            loadOutsideAirQuality();
            setInterval(loadOutsideAirQuality, 300000); // Update outside air quality every 5 minutes
        }, 1000); // Wait a bit for chart.js to be ready
        loadSchedule();
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const powerEl = document.getElementById('power-level');
                    if (powerEl && data.current_percent !== null && data.current_percent !== undefined) {
                        powerEl.textContent = Math.round(data.current_percent) + '%';
                    }
                    const aqi = data.last_aqi !== null ? Math.round(data.last_aqi) : null;
                    const aqiEl = document.getElementById('aqi');
                    if (aqiEl) {
                        aqiEl.textContent = aqi !== null ? aqi : '--';
                    }
                    const temperatureEl = document.getElementById('temperature');
                    if (temperatureEl) {
                        temperatureEl.innerHTML = data.last_temperature !== null ? data.last_temperature.toFixed(1) + '&deg;C' : '--';
                    }
                    const humidityEl = document.getElementById('humidity');
                    if (humidityEl) {
                        humidityEl.textContent = data.last_humidity !== null ? Math.round(data.last_humidity) + '%' : '--';
                    }
                    currentMode = data.mode || currentMode;
                    updateMenuItems();
                    updateAqiLabel(aqi);
                    updateModePill();
                })
                .catch(err => console.error('Status update failed:', err));
        }

        // Weather icon mapping
        function getWeatherIcon(weatherCode) {
            const icons = {
                0: '&#9728;',
                1: '&#9925;',
                2: '&#9925;',
                3: '&#9729;',
                45: '&#9729;',
                48: '&#9729;',
                51: '&#9748;',
                53: '&#9748;',
                55: '&#9748;',
                56: '&#9748;',
                57: '&#9748;',
                61: '&#9730;',
                63: '&#9730;',
                65: '&#9730;',
                66: '&#9730;',
                67: '&#9730;',
                71: '&#10052;',
                73: '&#10052;',
                75: '&#10052;',
                77: '&#10052;',
                80: '&#9730;',
                81: '&#9730;',
                82: '&#9730;',
                85: '&#10052;',
                86: '&#10052;',
                95: '&#9928;',
                96: '&#9928;',
                99: '&#9928;'
            };
            return icons[weatherCode] || '&#9729;';
        }

        function loadWeather() {
            const weatherContent = document.getElementById('weather-content');
            if (!weatherContent) return;

            fetch('/api/weather/belgrade')
                .then(r => r.json())
                .then(weather => {
                    if (weather && !weather.error) {
                        const currentIcon = getWeatherIcon(weather.current.weather_code);
                        let html = `
                            <div class="weather-current">
                                <div>
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                                        <span style="font-size: 28px;">${currentIcon}</span>
                                        <div class="weather-temp">${weather.current.temperature ? weather.current.temperature.toFixed(1) + '&deg;C' : '--'}</div>
                                    </div>
                                    <div class="weather-condition">${weather.current.condition || '--'}</div>
                                    <div class="weather-condition" style="font-size: 11px; margin-top: 3px;">Humidity: ${weather.current.humidity ? weather.current.humidity + '%' : '--'}</div>
                                </div>
                            </div>
                            <div class="weather-forecast">
                                ${weather.forecast ? weather.forecast.map(day => {
                                    const icon = getWeatherIcon(day.weather_code);
                                    return `
                                        <div class="forecast-day">
                                            <div class="forecast-date">${new Date(day.date).toLocaleDateString('en-US', {month: 'short', day: 'numeric'})}</div>
                                            <div style="font-size: 18px; margin: 4px 0;">${icon}</div>
                                            <div class="forecast-temp">${day.temp_max ? day.temp_max.toFixed(0) + '&deg;' : '--'}</div>
                                            <div class="forecast-min">${day.temp_min ? day.temp_min.toFixed(0) + '&deg;' : '--'}</div>
                                            <div class="forecast-min" style="font-size: 9px; margin-top: 3px;">${day.condition || '--'}</div>
                                        </div>
                                    `;
                                }).join('') : ''}
                            </div>
                        `;
                        weatherContent.innerHTML = html;
                    }
                })
                .catch(err => console.error('Weather load failed:', err));
        }

        function loadHistorical() {
            fetch('/api/historical')
                .then(r => r.json())
                .then(data => {
                    if (!data.data || data.data.length === 0) return;

                    // Process data for charts
                    const labels = data.data.map(d => {
                        const date = new Date(d.timestamp);
                        return date.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                    });

                    const aqiData = data.data.map(d => d.aqi || 0);
                    const tempData = data.data.map(d => d.temperature || 0);
                    const humData = data.data.map(d => d.humidity || 0);
                    const theme = getChartTheme();

                    // AQI Chart
                    const ctxAQI = document.getElementById('historicalChartAQI');
                    if (ctxAQI) {
                        if (historicalChartAQI) historicalChartAQI.destroy();
                        historicalChartAQI = new Chart(ctxAQI, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'AQI',
                                    data: aqiData,
                                    borderColor: theme.aqi,
                                    backgroundColor: toRgba(theme.aqi, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        beginAtZero: true,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }

                    // Temperature Chart
                    const ctxTemp = document.getElementById('historicalChartTemp');
                    if (ctxTemp) {
                        if (historicalChartTemp) historicalChartTemp.destroy();
                        historicalChartTemp = new Chart(ctxTemp, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Temperature (C)',
                                    data: tempData,
                                    borderColor: theme.temp,
                                    backgroundColor: toRgba(theme.temp, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        beginAtZero: false,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }

                    // Humidity Chart
                    const ctxHumidity = document.getElementById('historicalChartHumidity');
                    if (ctxHumidity) {
                        if (historicalChartHumidity) historicalChartHumidity.destroy();
                        historicalChartHumidity = new Chart(ctxHumidity, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Humidity (%)',
                                    data: humData,
                                    borderColor: theme.humidity,
                                    backgroundColor: toRgba(theme.humidity, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        min: 0,
                                        max: 100,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }

                    // Fan Speed Chart
                    const fanSpeedData = data.data.map(d => d.fan_speed || 0);
                    const ctxFanSpeed = document.getElementById('historicalChartFanSpeed');
                    if (ctxFanSpeed) {
                        if (historicalChartFanSpeed) historicalChartFanSpeed.destroy();
                        historicalChartFanSpeed = new Chart(ctxFanSpeed, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Fan Speed (%)',
                                    data: fanSpeedData,
                                    borderColor: theme.fan,
                                    backgroundColor: toRgba(theme.fan, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        min: 0,
                                        max: 100,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }
                })
                .catch(err => console.error('Historical data failed:', err));
        }

        function loadOutsideAQI() {
            fetch('/api/outside-aqi')
                .then(r => r.json())
                .then(data => {
                    if (data && !data.error) {
                        document.getElementById('outside-aqi').textContent = data.aqi !== null ? data.aqi : '--';
                        const pmEl = document.getElementById('outside-pm25');
                        if (pmEl) {
                            pmEl.innerHTML = data.pm25 !== null ? data.pm25.toFixed(1) + ' &micro;g/m&sup3;' : '--';
                        }
                    }
                })
                .catch(err => console.error('Outside AQI failed:', err));
        }

        function loadOutsideAirQuality() {
            fetch('/api/outside-air-quality')
                .then(r => r.json())
                .then(data => {
                    if (!data.data || data.data.length === 0) return;

                    const theme = getChartTheme();
                    const cityName = data.city || 'Outside';

                    // Update the section title with city name
                    const titleEl = document.querySelector('#outside-air-content-wrapper').closest('.graph-section').querySelector('.collapsible-title');
                    if (titleEl) {
                        titleEl.textContent = cityName + ' Outside Air Quality';
                    }

                    const labels = data.data.map(d => {
                        const date = new Date(d.timestamp);
                        return date.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                    });

                    const aqiData = data.data.map(d => d.aqi || 0);
                    const tempData = data.data.map(d => d.temperature || 0);

                    // Outside AQI Chart
                    const ctxAQI = document.getElementById('outsideAQIChart');
                    if (ctxAQI) {
                        if (outsideAQIChart) outsideAQIChart.destroy();
                        outsideAQIChart = new Chart(ctxAQI, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: cityName + ' AQI (24h)',
                                    data: aqiData,
                                    borderColor: theme.aqi,
                                    backgroundColor: toRgba(theme.aqi, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        beginAtZero: true,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }

                    // Outside Temperature Chart
                    const ctxTemp = document.getElementById('outsideTempChart');
                    if (ctxTemp) {
                        if (outsideTempChart) outsideTempChart.destroy();
                        outsideTempChart = new Chart(ctxTemp, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Outside Temperature (24h)',
                                    data: tempData,
                                    borderColor: theme.temp,
                                    backgroundColor: toRgba(theme.temp, 0.12),
                                    borderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {color: theme.text, font: {size: 11}}
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    },
                                    y: {
                                        beginAtZero: false,
                                        ticks: {color: theme.text, font: {size: 10}},
                                        grid: {color: theme.grid}
                                    }
                                }
                            }
                        });
                    }
                })
                .catch(err => console.error('Outside air quality failed:', err));
        }

        function loadSchedule() {
            fetch('/api/schedule')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('schedule-enabled').checked = data.enabled || false;
                    document.getElementById('schedule-start').value = data.start_hour || 22;
                    document.getElementById('schedule-end').value = data.end_hour || 7;
                    document.getElementById('schedule-post-mode').value = data.post_sleep_mode || 'auto';
                })
                .catch(err => console.error('Schedule load failed:', err));
        }
        
        function updateSchedule() {
            const enabled = document.getElementById('schedule-enabled').checked;
            const start = parseInt(document.getElementById('schedule-start').value);
            const end = parseInt(document.getElementById('schedule-end').value);
            const postMode = document.getElementById('schedule-post-mode').value;
            
            fetch('/api/schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    enabled: enabled,
                    start_hour: start,
                    end_hour: end,
                    post_sleep_mode: postMode
                })
            })
            .then(r => r.json())
            .then(data => {
                showToast('Schedule updated', 'success');
            })
            .catch(err => showToast('Failed to update schedule', 'error'));
        }
        
        function updateMenuItems() {
            document.querySelectorAll('.menu-item').forEach(item => {
                const isActive = item.dataset.mode === currentMode;
                item.classList.toggle('active', isActive);
                item.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }
        
        function selectMode(mode) {
            fetch('/api/set-mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            })
            .then(r => r.json())
            .then(data => {
                currentMode = mode;
                updateMenuItems();
                updateModePill();
                showToast('Mode changed to ' + mode, 'success');
                // Open settings panel
                openSettings(mode);
            })
            .catch(err => showToast('Failed to set mode', 'error'));
        }
        
        function openSettings(mode) {
            const panel = document.getElementById('settings-panel');
            const title = document.getElementById('settings-title');
            const content = document.getElementById('settings-content');
            
            // Set title
            const titles = {
                'auto': 'Auto Mode Settings',
                'curve': 'Curve Editor',
                'constant': 'Constant Mode Settings',
                'sleep': 'Sleep Mode'
            };
            title.textContent = titles[mode] || 'Settings';
            
            // Load settings content
            if (mode === 'auto') {
                loadAutoSettings(content);
            } else if (mode === 'curve') {
                loadCurveSettings(content);
            } else if (mode === 'constant') {
                loadConstantSettings(content);
            } else if (mode === 'sleep') {
                content.innerHTML = '<p style="opacity: 0.8; text-align: center; padding: 40px;">Sleep mode is active. The purifier is running at the lowest possible RPM.</p>';
            }
            
            // Panel is always visible, just update content
        }
        
        function closeSettings() {
            // Not needed in this layout, but keep for compatibility
            if (chart) {
                chart.destroy();
                chart = null;
            }
        }
        
        // Initialize - show default or current mode settings
        if (currentMode && document.getElementById('settings-panel')) {
            openSettings(currentMode);
        }
        
        // Auto mode settings
        function loadAutoSettings(container) {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const maxPercent = data.auto_max_percent || 100;
                    const curveType = data.auto_curve_type || 'linear';
                    container.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Maximum Power (%)</label>
                            <div class="input-group">
                                <input type="range" class="slider" id="auto-max-slider" min="10" max="100" value="${maxPercent}" oninput="updateAutoMax(this.value)">
                                <input type="number" class="number-input" id="auto-max-input" min="10" max="100" value="${maxPercent}" onchange="updateAutoMax(this.value)">
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Curve Type</label>
                            <select class="select-input" id="auto-curve-type">
                                <option value="linear" ${curveType === 'linear' ? 'selected' : ''}>Linear</option>
                                <option value="exponential" ${curveType === 'exponential' ? 'selected' : ''}>Exponential</option>
                                <option value="logarithmic" ${curveType === 'logarithmic' ? 'selected' : ''}>Logarithmic</option>
                                <option value="quadratic" ${curveType === 'quadratic' ? 'selected' : ''}>Quadratic</option>
                            </select>
                        </div>
                        <button class="btn" onclick="saveAutoSettings()">Save Settings</button>
                    `;
                });
        }
        
        function updateAutoMax(value) {
            document.getElementById('auto-max-slider').value = value;
            document.getElementById('auto-max-input').value = value;
        }
        
        function saveAutoSettings() {
            const maxPercent = parseInt(document.getElementById('auto-max-input').value);
            const curveType = document.getElementById('auto-curve-type').value;
            fetch('/api/set-auto-settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({max_percent: maxPercent, curve_type: curveType})
            })
            .then(r => r.json())
            .then(data => {
                showToast('Auto settings saved', 'success');
            })
            .catch(err => showToast('Failed to save settings', 'error'));
        }
        
        // Constant mode settings
        function loadConstantSettings(container) {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const percent = data.constant_percent || 50;
                    container.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Power Level (%)</label>
                            <div class="input-group">
                                <input type="range" class="slider" id="constant-slider" min="0" max="100" value="${percent}" oninput="updateConstantPercent(this.value)">
                                <input type="number" class="number-input" id="constant-input" min="0" max="100" value="${percent}" onchange="updateConstantPercent(this.value)">
                            </div>
                        </div>
                        <button class="btn" onclick="saveConstantSettings()">Save Settings</button>
                    `;
                });
        }
        
        function updateConstantPercent(value) {
            document.getElementById('constant-slider').value = value;
            document.getElementById('constant-input').value = value;
        }
        
        function saveConstantSettings() {
            const percent = parseInt(document.getElementById('constant-input').value);
            fetch('/api/set-constant-percent', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({percent: percent})
            })
            .then(r => r.json())
            .then(data => {
                showToast('Constant mode set to ' + percent + '%', 'success');
            })
            .catch(err => showToast('Failed to save settings', 'error'));
        }
        
        // Curve editor settings
        function loadCurveSettings(container) {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (data.current_curve) {
                        curvePoints = data.current_curve;
                    }
                    const smooth = data.curve_smooth || 'smooth';
                    const selectedCurve = data.selected_curve_name || 'Silent';
                    container.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Selected Curve (Active in Curve Mode)</label>
                            <select class="select-input" id="selected-curve" onchange="setSelectedCurve()">
                                <option value="">-- Select Curve --</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Smooth Algorithm</label>
                            <select class="select-input" id="curve-smooth" onchange="updateCurveSmooth()">
                                <option value="none" ${smooth === 'none' ? 'selected' : ''}>None (Straight Lines)</option>
                                <option value="smooth" ${smooth === 'smooth' ? 'selected' : ''}>Smooth</option>
                                <option value="bezier" ${smooth === 'bezier' ? 'selected' : ''}>Bezier</option>
                                <option value="catmull-rom" ${smooth === 'catmull-rom' ? 'selected' : ''}>Catmull-Rom</option>
                            </select>
                        </div>
                        <div class="chart-container">
                            <canvas id="curveChart"></canvas>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Add Point</label>
                            <div class="input-group">
                                <input type="number" class="point-input" id="new-aqi" placeholder="AQI" min="0" max="600">
                                <input type="number" class="point-input" id="new-percent" placeholder="%" min="0" max="100">
                                <button class="btn btn-small" onclick="addPointManual()">Add</button>
                            </div>
                        </div>
                        <div class="point-list" id="point-list"></div>
                        <div class="form-group">
                            <label class="form-label">Save Curve</label>
                            <div class="input-group">
                                <input type="text" class="point-input" id="curve-name" placeholder="Curve name" style="flex: 1;">
                                <button class="btn btn-small" onclick="saveCurve()">Save</button>
                            </div>
                        </div>
                        <div class="saved-curves" id="saved-curves"></div>
                    `;
                    setTimeout(() => {
                        initCurveEditor();
                        updatePointList();
                        loadSavedCurves();
                        populateCurveSelector(selectedCurve);
                    }, 100);
                });
        }
        
        function populateCurveSelector(selected) {
            const select = document.getElementById('selected-curve');
            if (!select) return;
            
            fetch('/api/list-curves')
                .then(r => r.json())
                .then(data => {
                    const curves = Object.keys(data.curves || {});
                    select.innerHTML = '<option value="">-- Select Curve --</option>' +
                        curves.map(name => `<option value="${name}" ${name === selected ? 'selected' : ''}>${name}</option>`).join('');
                });
        }
        
        function setSelectedCurve() {
            const select = document.getElementById('selected-curve');
            const curveName = select.value;
            if (!curveName) return;
            
            fetch('/api/set-selected-curve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({curve_name: curveName})
            })
            .then(r => r.json())
            .then(data => {
                showToast('Selected curve: ' + curveName, 'success');
                // Reload curve editor
                if (currentMode === 'curve') {
                    loadCurve(curveName);
                }
            })
            .catch(err => showToast('Failed to set curve', 'error'));
        }
        
        function initCurveEditor() {
            if (chart) chart.destroy();
            const ctx = document.getElementById('curveChart').getContext('2d');
            
            // Get current smooth algorithm setting
            const smoothSelect = document.getElementById('curve-smooth');
            const smooth = smoothSelect ? smoothSelect.value : 'smooth';
            
            // Determine tension and interpolation mode based on algorithm
            let tension = 0.4;
            let cubicInterpolationMode = 'default';
            if (smooth === 'none') {
                tension = 0;
                cubicInterpolationMode = 'default';
            } else if (smooth === 'smooth') {
                tension = 0.4;
                cubicInterpolationMode = 'default';
            } else if (smooth === 'bezier') {
                tension = 0.5;
                cubicInterpolationMode = 'default';
            } else if (smooth === 'catmull-rom') {
                tension = 0.5;
                cubicInterpolationMode = 'monotone';
            }
            
            const theme = getChartTheme();
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Fan Speed %',
                        data: curvePoints.map(p => ({x: p[0], y: p[1]})),
                        borderColor: theme.temp,
                        backgroundColor: toRgba(theme.temp, 0.12),
                        borderWidth: 2,
                        pointRadius: 6,
                        pointBackgroundColor: theme.aqi,
                        pointBorderColor: theme.text,
                        pointHoverRadius: 8,
                        tension: tension,
                        cubicInterpolationMode: cubicInterpolationMode
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            min: 0,
                            max: 600,
                            title: {display: true, text: 'AQI', color: theme.text},
                            ticks: {color: theme.text},
                            grid: {color: theme.grid}
                        },
                        y: {
                            min: 0,
                            max: 100,
                            title: {display: true, text: 'Power %', color: theme.text},
                            ticks: {color: theme.text},
                            grid: {color: theme.grid}
                        }
                    },
                    onClick: (e) => {
                        const canvas = e.native.target;
                        const rect = canvas.getBoundingClientRect();
                        const x = e.native.offsetX;
                        const y = e.native.offsetY;
                        const xScale = chart.scales.x;
                        const yScale = chart.scales.y;
                        const aqi = xScale.getValueForPixel(x);
                        const percent = 100 - ((y - yScale.top) / (yScale.bottom - yScale.top)) * 100;
                        if (aqi >= 0 && aqi <= 600 && percent >= 0 && percent <= 100) {
                            addPoint(Math.round(aqi), Math.round(percent));
                        }
                    }
                }
            });
        }
        
        function addPoint(aqi, percent) {
            curvePoints.push([aqi, percent]);
            curvePoints.sort((a, b) => a[0] - b[0]);
            updateChart();
            updatePointList();
        }
        
        function addPointManual() {
            const aqi = parseInt(document.getElementById('new-aqi').value);
            const percent = parseInt(document.getElementById('new-percent').value);
            if (!isNaN(aqi) && !isNaN(percent) && aqi >= 0 && aqi <= 600 && percent >= 0 && percent <= 100) {
                addPoint(aqi, percent);
                document.getElementById('new-aqi').value = '';
                document.getElementById('new-percent').value = '';
            }
        }
        
        function updateChart() {
            if (!chart) return;
            
            // Update all data points
            chart.data.datasets[0].data = curvePoints.map(p => ({x: p[0], y: p[1]}));
            const theme = getChartTheme();
            chart.data.datasets[0].borderColor = theme.temp;
            chart.data.datasets[0].backgroundColor = toRgba(theme.temp, 0.12);
            chart.data.datasets[0].pointBackgroundColor = theme.aqi;
            chart.data.datasets[0].pointBorderColor = theme.text;
            chart.options.scales.x.ticks.color = theme.text;
            chart.options.scales.y.ticks.color = theme.text;
            chart.options.scales.x.title.color = theme.text;
            chart.options.scales.y.title.color = theme.text;
            chart.options.scales.x.grid.color = theme.grid;
            chart.options.scales.y.grid.color = theme.grid;
            
            // Get current smooth algorithm and apply to ALL connections between ALL dots
            const smoothSelect = document.getElementById('curve-smooth');
            const smooth = smoothSelect ? smoothSelect.value : 'smooth';
            
            // Apply smooth algorithm to all connections between dots
            if (smooth === 'none') {
                chart.data.datasets[0].tension = 0; // Straight lines between all points
                chart.data.datasets[0].cubicInterpolationMode = 'default';
            } else if (smooth === 'smooth') {
                chart.data.datasets[0].tension = 0.4; // Smooth curves between all points
                chart.data.datasets[0].cubicInterpolationMode = 'default';
            } else if (smooth === 'bezier') {
                chart.data.datasets[0].tension = 0.5; // Bezier-like curves between all points
                chart.data.datasets[0].cubicInterpolationMode = 'default';
            } else if (smooth === 'catmull-rom') {
                chart.data.datasets[0].tension = 0.5; // Catmull-Rom style between all points
                chart.data.datasets[0].cubicInterpolationMode = 'monotone';
            } else {
                chart.data.datasets[0].tension = 0.4;
                chart.data.datasets[0].cubicInterpolationMode = 'default';
            }
            
            // Force full chart redraw to apply algorithm to ALL connections
            chart.update('active'); // Update with animation to show the change
            
            // Apply curve to server if in curve mode
            if (currentMode === 'curve' && curvePoints.length >= 2) {
                applyCurve();
            }
        }
        
        function applyCurve() {
            const smooth = document.getElementById('curve-smooth').value;
            fetch('/api/set-curve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({curve: curvePoints, smooth: smooth, name: 'Current'})
            })
            .catch(err => console.error('Failed to apply curve:', err));
        }
        
        function updatePointList() {
            const list = document.getElementById('point-list');
            if (!list) return;
            list.innerHTML = curvePoints.map((point, idx) => `
                <div class="point-item">
                    <input type="number" class="point-input" value="${point[0]}" onchange="updatePoint(${idx}, 0, this.value)" placeholder="AQI">
                    <input type="number" class="point-input" value="${point[1]}" onchange="updatePoint(${idx}, 1, this.value)" placeholder="%">
                    <button class="delete-point" onclick="deletePoint(${idx})">x</button>
                </div>
            `).join('');
        }
        
        function updatePoint(idx, coord, value) {
            curvePoints[idx][coord] = parseFloat(value) || 0;
            curvePoints.sort((a, b) => a[0] - b[0]);
            updateChart();
            updatePointList();
        }
        
        function deletePoint(idx) {
            if (curvePoints.length > 2) {
                curvePoints.splice(idx, 1);
                updateChart();
                updatePointList();
            }
        }
        
        function updateCurveSmooth() {
            // When smooth algorithm changes, reapply to all connections
            if (chart && curvePoints.length >= 2) {
                updateChart();
            }
        }
        
        function saveCurve() {
            const name = document.getElementById('curve-name').value.trim();
            if (!name) {
                showToast('Please enter a curve name', 'error');
                return;
            }
            if (curvePoints.length < 2) {
                showToast('At least 2 points required', 'error');
                return;
            }
            const smooth = document.getElementById('curve-smooth').value;
            fetch('/api/save-curve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, curve: curvePoints, smooth: smooth})
            })
            .then(r => r.json())
            .then(data => {
                showToast('Curve saved', 'success');
                loadSavedCurves();
                document.getElementById('curve-name').value = '';
            })
            .catch(err => showToast('Failed to save curve', 'error'));
        }
        
        function loadSavedCurves() {
            fetch('/api/list-curves')
                .then(r => r.json())
                .then(data => {
                    savedCurvesList = data.curves || {};
                    const container = document.getElementById('saved-curves');
                    if (!container) return;
                    const curves = Object.keys(savedCurvesList);
                    if (curves.length === 0) {
                        container.innerHTML = '<div style="text-align: center; opacity: 0.6; padding: 20px; font-size: 14px;">No saved curves</div>';
                        return;
                    }
                    container.innerHTML = curves.map(name => `
                        <div class="curve-item">
                            <div class="curve-item-name">${name}</div>
                            <div class="curve-item-actions">
                                <button class="btn btn-small btn-secondary" onclick="loadCurve('${name}')">Load</button>
                                <button class="btn btn-small btn-danger" onclick="deleteCurve('${name}')">Delete</button>
                            </div>
                        </div>
                    `).join('');
                })
                .catch(err => console.error('Failed to load curves:', err));
        }
        
        function loadCurve(name) {
            fetch('/api/load-curve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name})
            })
            .then(r => r.json())
            .then(data => {
                curvePoints = data.current_curve || curvePoints;
                document.getElementById('curve-smooth').value = data.curve_smooth || 'smooth';
                updateChart();
                updatePointList();
                showToast('Curve loaded', 'success');
            })
            .catch(err => showToast('Failed to load curve', 'error'));
        }
        
        function deleteCurve(name) {
            if (!confirm('Delete curve "' + name + '"?')) return;
            fetch('/api/delete-curve', {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name})
            })
            .then(r => r.json())
            .then(data => {
                showToast('Curve deleted', 'success');
                loadSavedCurves();
            })
            .catch(err => showToast('Failed to delete curve', 'error'));
        }
        
        function showToast(message, type = '') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast ' + type + ' show';
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        // ============= Logging Functions =============
        let loggingMinInterval = 10;

        function loadLoggingStatus() {
            fetch('/api/logging/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('logging-enabled').checked = data.enabled;
                    document.getElementById('log-interval-slider').value = data.interval;
                    document.getElementById('log-interval-number').value = data.interval;
                    loggingMinInterval = data.min_interval || 10;
                    document.getElementById('log-interval-slider').min = loggingMinInterval;
                    document.getElementById('log-interval-number').min = loggingMinInterval;
                })
                .catch(err => console.error('Error loading logging status:', err));
        }

        function updateLogging() {
            const enabled = document.getElementById('logging-enabled').checked;
            fetch('/api/logging/enable', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: enabled })
            })
            .then(r => r.json())
            .then(data => {
                showToast(data.enabled ? 'Logging enabled' : 'Logging disabled', 'success');
            })
            .catch(err => showToast('Failed to update logging', 'error'));
        }

        function updateLogInterval(value) {
            value = Math.max(loggingMinInterval, parseInt(value) || loggingMinInterval);
            document.getElementById('log-interval-slider').value = value;
            document.getElementById('log-interval-number').value = value;

            fetch('/api/logging/interval', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interval: value })
            })
            .then(r => r.json())
            .catch(err => console.error('Error updating interval:', err));
        }

        // ============= Temperature Assist Functions =============
        function loadTempAssistStatus() {
            fetch('/api/temp-assist/status')
                .then(r => r.json())
                .then(data => {
                    const enabledEl = document.getElementById('temp-assist-enabled');
                    const thresholdEl = document.getElementById('temp-assist-threshold');
                    const reduceSlider = document.getElementById('temp-assist-reduce-slider');
                    const reduceNumber = document.getElementById('temp-assist-reduce-number');

                    if (enabledEl) enabledEl.checked = !!data.enabled;
                    if (thresholdEl) thresholdEl.value = (data.threshold_c ?? 18.0);
                    if (reduceSlider) reduceSlider.value = (data.reduce_percent ?? 10);
                    if (reduceNumber) reduceNumber.value = (data.reduce_percent ?? 10);
                })
                .catch(err => console.error('Error loading temp assist status:', err));
        }

        function syncTempAssistReduce(value) {
            const v = Math.max(0, Math.min(100, parseFloat(value) || 0));
            const reduceSlider = document.getElementById('temp-assist-reduce-slider');
            const reduceNumber = document.getElementById('temp-assist-reduce-number');
            if (reduceSlider) reduceSlider.value = v;
            if (reduceNumber) reduceNumber.value = v;
            updateTempAssist();
        }

        function updateTempAssist() {
            const enabled = document.getElementById('temp-assist-enabled')?.checked || false;
            const thresholdC = parseFloat(document.getElementById('temp-assist-threshold')?.value);
            const reducePercent = parseFloat(document.getElementById('temp-assist-reduce-number')?.value);

            fetch('/api/temp-assist/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    enabled: enabled,
                    threshold_c: isFinite(thresholdC) ? thresholdC : 18.0,
                    reduce_percent: isFinite(reducePercent) ? reducePercent : 10
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data && data.error) showToast(data.error, 'error');
            })
            .catch(err => showToast('Failed to update Temperature Assist', 'error'));
        }

        // ============= Log Explorer Functions =============
        let currentLogPreview = null;

        function openLogExplorer() {
            document.getElementById('log-explorer-modal').classList.add('show');
            loadLogTree();
            document.addEventListener('keydown', logExplorerKeyHandler);
        }

        function closeLogExplorer() {
            document.getElementById('log-explorer-modal').classList.remove('show');
            document.removeEventListener('keydown', logExplorerKeyHandler);
        }

        function logExplorerKeyHandler(e) {
            if (e.key === 'Escape') closeLogExplorer();
        }

        function loadLogTree() {
            const body = document.getElementById('log-explorer-body');
            body.innerHTML = '<div style="text-align: center; color: var(--muted);">Loading...</div>';

            fetch('/api/logs/tree')
                .then(r => r.json())
                .then(data => {
                    if (Object.keys(data).length === 0) {
                        body.innerHTML = '<div style="text-align: center; color: var(--muted); padding: 40px;">No log files yet. Enable logging and wait for data to be collected.</div>';
                        return;
                    }

                    let html = '';
                    const folderNames = {
                        'aqi': 'Indoor AQI',
                        'temperature': 'Temperature',
                        'humidity': 'Humidity',
                        'fan_speed': 'Fan Speed',
                        'outside_aqi': 'Outside AQI',
                        'outside_temp': 'Outside Temp'
                    };

                    for (const [folder, files] of Object.entries(data)) {
                        const totalRows = files.reduce((sum, f) => sum + f.rows, 0);
                        html += `
                            <div class="log-folder" id="folder-${folder}">
                                <div class="log-folder-header" onclick="toggleLogFolder('${folder}')">
                                    <span class="log-folder-icon">&#128193;</span>
                                    <span class="log-folder-name">${folderNames[folder] || folder}</span>
                                    <span class="log-folder-count">${files.length} files, ${totalRows.toLocaleString()} rows</span>
                                </div>
                                <div class="log-folder-files">
                        `;

                        for (const file of files) {
                            const sizeKB = (file.size / 1024).toFixed(1);
                            html += `
                                <div class="log-file">
                                    <span class="log-file-icon">&#128196;</span>
                                    <div class="log-file-info">
                                        <div class="log-file-name">${file.name}</div>
                                        <div class="log-file-meta">${sizeKB} KB | ${file.rows.toLocaleString()} rows</div>
                                    </div>
                                    <div class="log-file-actions">
                                        <button class="log-file-btn" onclick="previewLogFile('${file.path}')">Preview</button>
                                        <a class="log-file-btn" href="/api/logs/download/${file.path}" download>Download</a>
                                    </div>
                                </div>
                            `;
                        }

                        html += '</div></div>';
                    }

                    html += '<div id="log-preview-area"></div>';
                    body.innerHTML = html;
                })
                .catch(err => {
                    body.innerHTML = '<div style="text-align: center; color: var(--muted);">Error loading log files</div>';
                });
        }

        function toggleLogFolder(folder) {
            const elem = document.getElementById('folder-' + folder);
            elem.classList.toggle('expanded');
        }

        function previewLogFile(path) {
            const area = document.getElementById('log-preview-area');
            area.innerHTML = '<div style="text-align: center; color: var(--muted); padding: 20px;">Loading preview...</div>';

            fetch('/api/logs/file/' + path)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        area.innerHTML = '<div class="log-preview">Error: ' + data.error + '</div>';
                        return;
                    }

                    let content = data.header + '\\n' + data.preview.join('\\n');
                    area.innerHTML = `
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid var(--border);">
                            <div style="font-size: 13px; margin-bottom: 8px; color: var(--ink);">
                                <strong>Preview:</strong> Showing ${data.showing} of ${data.total_rows} rows
                            </div>
                            <div class="log-preview">${content}</div>
                        </div>
                    `;
                })
                .catch(err => {
                    area.innerHTML = '<div class="log-preview">Error loading preview</div>';
                });
        }

        // ============= Analysis Functions =============
        let selectedWeekdays = [];
        let generatedCurve = null;
        let curvePreviewChart = null;

        function loadAnalysisDataRange() {
            fetch('/api/analysis/data-range')
                .then(r => r.json())
                .then(data => {
                    const aqiHours = data.aqi?.hours || 0;
                    const fanHours = data.fan_speed?.hours || 0;

                    document.getElementById('analysis-aqi-hours').textContent =
                        aqiHours > 0 ? `${aqiHours.toFixed(1)} hours (${(data.aqi?.rows || 0).toLocaleString()} points)` : 'No data';
                    document.getElementById('analysis-fan-hours').textContent =
                        fanHours > 0 ? `${fanHours.toFixed(1)} hours (${(data.fan_speed?.rows || 0).toLocaleString()} points)` : 'No data';
                })
                .catch(err => console.error('Error loading data range:', err));
        }

        function toggleWeekday(day) {
            const btn = document.querySelector(`.weekday-btn[data-day="${day}"]`);
            const idx = selectedWeekdays.indexOf(day);

            if (idx === -1) {
                selectedWeekdays.push(day);
                btn.classList.add('active');
            } else {
                selectedWeekdays.splice(idx, 1);
                btn.classList.remove('active');
            }
        }

        function generateAnalysisCurve() {
            const hours = parseInt(document.getElementById('analysis-period').value);
            const includeOutsideAqi = document.getElementById('analysis-outside-aqi').checked;
            const maxCutoff = parseInt(document.getElementById('analysis-cutoff').value) / 100;
            const weekdays = selectedWeekdays.length > 0 ? selectedWeekdays : null;

            showToast('Generating curve...', '');

            fetch('/api/analysis/generate-curve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    hours: hours,
                    include_outside_aqi: includeOutsideAqi,
                    max_cutoff: maxCutoff,
                    weekdays: weekdays
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showToast('Error: ' + data.error, 'error');
                    return;
                }

                generatedCurve = data.curve;
                showCurvePreview(data);
                showToast('Curve generated successfully', 'success');
            })
            .catch(err => showToast('Failed to generate curve', 'error'));
        }

        function showCurvePreview(data) {
            const container = document.getElementById('curve-preview-container');
            const info = document.getElementById('curve-preview-info');

            info.textContent = `Generated from ${data.data_points.toLocaleString()} data points over ${data.hours_analyzed} hours using ${data.buckets_used} AQI buckets`;

            container.style.display = 'block';

            // Create or update chart
            const ctx = document.getElementById('curve-preview-canvas').getContext('2d');
            const theme = getChartTheme();

            if (curvePreviewChart) {
                curvePreviewChart.destroy();
            }

            curvePreviewChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.curve.map(p => p[0]),
                    datasets: [{
                        label: 'Fan Speed %',
                        data: data.curve.map(p => p[1]),
                        borderColor: theme.accent,
                        backgroundColor: theme.accentSoft,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: { display: true, text: 'AQI', color: theme.text },
                            ticks: { color: theme.text },
                            grid: { color: theme.grid }
                        },
                        y: {
                            title: { display: true, text: 'Fan Speed %', color: theme.text },
                            min: 0,
                            max: 100,
                            ticks: { color: theme.text },
                            grid: { color: theme.grid }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        function loadGeneratedCurve() {
            if (!generatedCurve || generatedCurve.length < 2) {
                showToast('No curve to load', 'error');
                return;
            }

            const curveName = document.getElementById('analysis-curve-name').value || 'Generated Curve';

            // Save the curve
            fetch('/api/save-curve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: curveName,
                    curve: generatedCurve,
                    smooth: 'smooth'
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast('Curve saved as "' + curveName + '"', 'success');
                    // Open curve editor
                    selectMode('curve');
                    setTimeout(() => openSettings('curve'), 100);
                } else {
                    showToast('Failed to save curve', 'error');
                }
            })
            .catch(err => showToast('Failed to save curve', 'error'));
        }

        // Initialize logging and analysis on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadLoggingStatus();
            loadAnalysisDataRange();
            loadTempAssistStatus();
        });

    </script>
</body>
</html>


"""


def snapshot_state() -> dict:
    with state_lock:
        return dict(state)


@app.get("/")
def index():
    """Main dashboard page"""
    snap = snapshot_state()
    return render_template_string(HTML_TEMPLATE, state=snap, page="purifier")


@app.get("/api/status")
def api_status():
    """Get current status"""
    return jsonify(snapshot_state())


@app.post("/api/set-mode")
def api_set_mode():
    """Set the purifier mode"""
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "")).strip().lower()
    
    if mode not in ["auto", "curve", "constant", "sleep"]:
        return jsonify({"error": "Invalid mode"}), 400
    
    with state_lock:
        state["mode"] = mode
        state["last_mode_change"] = dt.datetime.now().isoformat(timespec="seconds")
        state["mode_change_in_progress"] = True
        
        # If switching to curve mode, load the selected curve
        if mode == "curve":
            selected_curve_name = state.get("selected_curve_name", "Silent")
            with curves_lock:
                if selected_curve_name in saved_curves:
                    curve_data = saved_curves[selected_curve_name]
                    state["current_curve"] = curve_data["points"]
                    state["current_curve_name"] = selected_curve_name
                    state["curve_smooth"] = curve_data.get("smooth", "smooth")
    
    # Save state to file
    save_state()
    
    # All modes use favorite mode on device, just control speed
    # Sleep mode: apply minimum immediately
    if mode == "sleep":
        try:
            client.apply_percent(int((1 / LEVEL_MAX) * 100))
        except:
            pass
    
    def clear_flag():
        time.sleep(3)
        with state_lock:
            state["mode_change_in_progress"] = False
    threading.Thread(target=clear_flag, daemon=True).start()
    
    return jsonify(snapshot_state())


@app.post("/api/set-auto-settings")
def api_set_auto_settings():
    """Set auto mode settings"""
    payload = request.get_json(silent=True) or {}
    
    with state_lock:
        if "max_percent" in payload:
            state["auto_max_percent"] = max(0, min(100, int(payload["max_percent"])))
        if "curve_type" in payload:
            curve_type = str(payload["curve_type"]).strip().lower()
            if curve_type in ["linear", "exponential", "logarithmic", "quadratic"]:
                state["auto_curve_type"] = curve_type
    
    save_state()
    return jsonify(snapshot_state())


@app.post("/api/set-constant-percent")
def api_set_constant_percent():
    """Set constant mode percentage"""
    payload = request.get_json(silent=True) or {}
    percent = payload.get("percent")
    
    if percent is None:
        return jsonify({"error": "percent required"}), 400
    
    try:
        percent = int(percent)
        percent = max(0, min(100, percent))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid percent"}), 400
    
    with state_lock:
        state["constant_percent"] = percent
    
    save_state()
    
    # Apply immediately if in constant mode
    with state_lock:
        if state["mode"] == "constant":
            client.apply_percent(percent)
    
    return jsonify(snapshot_state())


@app.post("/api/set-curve")
def api_set_curve():
    """Set current curve points"""
    payload = request.get_json(silent=True) or {}
    curve = payload.get("curve", [])
    smooth = payload.get("smooth", "smooth")
    name = payload.get("name", "Unnamed")
    
    if not isinstance(curve, list):
        return jsonify({"error": "curve must be a list"}), 400
    
    # Validate curve points
    validated_curve = []
    for point in curve:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            aqi = float(point[0])
            percent = float(point[1])
            aqi = max(0, min(600, aqi))  # AQI up to 600
            percent = max(0, min(100, percent))
            validated_curve.append([aqi, percent])
        except (TypeError, ValueError):
            continue
    
    if len(validated_curve) < 2:
        return jsonify({"error": "At least 2 points required"}), 400
    
    with state_lock:
        state["current_curve"] = validated_curve
        state["current_curve_name"] = name
        if smooth in ["none", "smooth", "bezier", "catmull-rom"]:
            state["curve_smooth"] = smooth
    
    save_state()
    return jsonify(snapshot_state())


@app.post("/api/save-curve")
def api_save_curve():
    """Save a curve with a name"""
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    curve = payload.get("curve", [])
    smooth = payload.get("smooth", "smooth")
    
    if not name:
        return jsonify({"error": "name required"}), 400
    
    if not isinstance(curve, list) or len(curve) < 2:
        return jsonify({"error": "valid curve required"}), 400
    
    # Validate and store
    validated_curve = []
    for point in curve:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            aqi = float(point[0])
            percent = float(point[1])
            validated_curve.append([max(0, min(600, aqi)), max(0, min(100, percent))])
        except (TypeError, ValueError):
            continue
    
    if len(validated_curve) < 2:
        return jsonify({"error": "At least 2 valid points required"}), 400
    
    with curves_lock:
        saved_curves[name] = {
            "points": validated_curve,
            "smooth": smooth if smooth in ["none", "smooth", "bezier", "catmull-rom"] else "smooth",
            "created": dt.datetime.now().isoformat()
        }
    
    return jsonify({"success": True, "saved_curves": list(saved_curves.keys())})


@app.get("/api/list-curves")
def api_list_curves():
    """List all saved curves"""
    with curves_lock:
        return jsonify({
            "curves": {name: {
                "points": curve["points"],
                "smooth": curve["smooth"],
                "created": curve["created"]
            } for name, curve in saved_curves.items()}
        })


@app.post("/api/load-curve")
def api_load_curve():
    """Load a saved curve"""
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    
    if not name:
        return jsonify({"error": "name required"}), 400
    
    with curves_lock:
        if name not in saved_curves:
            return jsonify({"error": "Curve not found"}), 404
        
        curve_data = saved_curves[name]
        with state_lock:
            state["current_curve"] = curve_data["points"]
            state["current_curve_name"] = name
            state["curve_smooth"] = curve_data["smooth"]
    
    save_state()
    return jsonify(snapshot_state())


@app.delete("/api/delete-curve")
def api_delete_curve():
    """Delete a saved curve"""
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    
    if not name:
        return jsonify({"error": "name required"}), 400
    
    with curves_lock:
        if name in saved_curves:
            del saved_curves[name]
            return jsonify({"success": True})
        return jsonify({"error": "Curve not found"}), 404

@app.get("/api/weather/<location>")
def api_weather(location: str):
    """Get weather data for a location"""
    weather = get_weather(location)
    return jsonify(weather)

@app.get("/api/historical")
def api_historical():
    """Get historical data for last 6 hours"""
    with historical_lock:
        # Filter to last 6 hours
        cutoff = dt.datetime.now() - dt.timedelta(hours=6)
        filtered = [
            d for d in historical_data
            if dt.datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        return jsonify({"data": filtered})

@app.get("/api/schedule")
def api_get_schedule():
    """Get sleep schedule settings"""
    return jsonify(sleep_schedule)

@app.post("/api/schedule")
def api_set_schedule():
    """Set sleep schedule settings"""
    payload = request.get_json(silent=True) or {}
    
    global sleep_schedule
    if "enabled" in payload:
        sleep_schedule["enabled"] = bool(payload["enabled"])
    if "start_hour" in payload:
        sleep_schedule["start_hour"] = max(0, min(23, int(payload["start_hour"])))
    if "end_hour" in payload:
        sleep_schedule["end_hour"] = max(0, min(23, int(payload["end_hour"])))
    if "post_sleep_mode" in payload:
        mode = str(payload["post_sleep_mode"]).strip().lower()
        if mode in ["auto", "curve", "constant"]:
            sleep_schedule["post_sleep_mode"] = mode
    
    save_schedule()
    return jsonify(sleep_schedule)

@app.get("/api/outside-aqi")
def api_outside_aqi():
    """Get outside AQI for configured location"""
    aqi_data = get_outside_aqi()
    return jsonify(aqi_data)

@app.get("/api/outside-air-quality")
def api_outside_air_quality():
    """Get outside air quality history for configured location"""
    history = get_outside_air_quality_history()
    return jsonify({"data": history, "city": AIR_QUALITY_CITY})

@app.post("/api/set-selected-curve")
def api_set_selected_curve():
    """Set the selected curve for curve mode"""
    payload = request.get_json(silent=True) or {}
    curve_name = str(payload.get("curve_name", "")).strip()
    
    if not curve_name:
        return jsonify({"error": "curve_name required"}), 400
    
    with curves_lock:
        if curve_name not in saved_curves:
            return jsonify({"error": "Curve not found"}), 404
        
        curve_data = saved_curves[curve_name]
        with state_lock:
            state["selected_curve_name"] = curve_name
            state["current_curve"] = curve_data["points"]
            state["current_curve_name"] = curve_name
            state["curve_smooth"] = curve_data["smooth"]
    
    save_state()
    return jsonify(snapshot_state())


# ============= Logging API Endpoints =============

@app.get("/api/logging/status")
def api_logging_status():
    """Get current logging status"""
    with state_lock:
        return jsonify({
            "enabled": state.get("logging_enabled", False),
            "interval": state.get("log_interval", MIN_LOG_INTERVAL),
            "min_interval": MIN_LOG_INTERVAL
        })


@app.post("/api/logging/enable")
def api_logging_enable():
    """Enable or disable logging"""
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled", False)

    with state_lock:
        state["logging_enabled"] = bool(enabled)

    save_state()
    return jsonify({
        "enabled": state.get("logging_enabled", False),
        "interval": state.get("log_interval", MIN_LOG_INTERVAL)
    })

# ============= Temperature Assist API Endpoints =============

@app.get("/api/temp-assist/status")
def api_temp_assist_status():
    """Get Temperature Assist configuration"""
    with state_lock:
        return jsonify({
            "enabled": bool(state.get("temp_assist_enabled", False)),
            "threshold_c": state.get("temp_assist_threshold_c", 18.0),
            "reduce_percent": state.get("temp_assist_reduce_percent", 10),
        })


@app.post("/api/temp-assist/config")
def api_temp_assist_config():
    """Update Temperature Assist configuration"""
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))
    threshold_c = payload.get("threshold_c", 18.0)
    reduce_percent = payload.get("reduce_percent", 10)

    try:
        threshold_c = float(threshold_c)
        reduce_percent = float(reduce_percent)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameters"}), 400

    threshold_c = max(-10.0, min(40.0, threshold_c))
    reduce_percent = max(0.0, min(100.0, reduce_percent))

    with state_lock:
        state["temp_assist_enabled"] = enabled
        state["temp_assist_threshold_c"] = threshold_c
        state["temp_assist_reduce_percent"] = reduce_percent

    save_state()
    return jsonify({
        "enabled": bool(state.get("temp_assist_enabled", False)),
        "threshold_c": state.get("temp_assist_threshold_c", 18.0),
        "reduce_percent": state.get("temp_assist_reduce_percent", 10),
    })


@app.post("/api/logging/interval")
def api_logging_interval():
    """Set logging interval"""
    payload = request.get_json(silent=True) or {}
    interval = payload.get("interval", MIN_LOG_INTERVAL)

    try:
        interval = max(MIN_LOG_INTERVAL, int(interval))
    except (TypeError, ValueError):
        interval = MIN_LOG_INTERVAL

    with state_lock:
        state["log_interval"] = interval

    save_state()
    return jsonify({
        "enabled": state.get("logging_enabled", False),
        "interval": state.get("log_interval", MIN_LOG_INTERVAL)
    })


# ============= Log Explorer API Endpoints =============

@app.get("/api/logs/tree")
def api_logs_tree():
    """Get log file tree structure"""
    return jsonify(log_manager.get_log_tree())


@app.get("/api/logs/file/<path:filepath>")
def api_logs_file_preview(filepath: str):
    """Preview a log file content"""
    preview = log_manager.read_file_preview(filepath)
    return jsonify(preview)


@app.get("/api/logs/download/<path:filepath>")
def api_logs_download(filepath: str):
    """Download a log file"""
    absolute_path = log_manager.get_file_path(filepath)
    if absolute_path is None:
        return jsonify({"error": "File not found"}), 404

    return send_file(
        absolute_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=os.path.basename(filepath)
    )


# ============= Analysis API Endpoints =============

@app.get("/api/analysis/data-range")
def api_analysis_data_range():
    """Get available data range for analysis"""
    return jsonify(analysis_engine.get_available_data_range())


@app.post("/api/analysis/generate-curve")
def api_analysis_generate_curve():
    """Generate an optimized curve from logged data"""
    payload = request.get_json(silent=True) or {}

    hours = payload.get("hours", 48)
    include_outside_aqi = payload.get("include_outside_aqi", False)
    max_cutoff = payload.get("max_cutoff", 1.0)
    weekdays = payload.get("weekdays", None)

    try:
        hours = max(1, int(hours))
        max_cutoff = max(0.2, min(1.0, float(max_cutoff)))
        if weekdays is not None:
            weekdays = [int(d) for d in weekdays if 0 <= int(d) <= 6]
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameters"}), 400

    result = analysis_engine.generate_curve(
        hours=hours,
        include_outside_aqi=include_outside_aqi,
        max_cutoff=max_cutoff,
        weekdays=weekdays
    )

    return jsonify(result)


# Start control loop in background
threading.Thread(target=control_loop, daemon=True).start()

# Start logging loop in background
threading.Thread(target=logging_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
