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
from flask import Flask, Response, jsonify, render_template_string, request

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

def get_belgrade_aqi() -> dict:
    """Get Belgrade AQI from OpenAQ API (free, no API key needed)"""
    try:
        # Try OpenAQ API for air quality
        url = "https://api.openaq.org/v2/latest?location_id=7961&limit=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                measurements = result.get("measurements", [])
                pm25 = None
                pm10 = None
                for m in measurements:
                    if m.get("parameter") == "pm25":
                        pm25 = m.get("value")
                    elif m.get("parameter") == "pm10":
                        pm10 = m.get("value")
                
                # Convert PM2.5 to AQI (simplified)
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
                    "location": "Belgrade",
                    "timestamp": result.get("lastUpdated", "")
                }
    except Exception as e:
        print(f"Belgrade AQI API error: {e}")
    
    # Fallback: try Open-Meteo air quality API
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=44.7866&longitude=20.4489&current=pm10,pm2_5&timezone=Europe/Belgrade"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            pm25 = current.get("pm2_5")
            pm10 = current.get("pm10")
            
            aqi = None
            if pm25:
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
                "location": "Belgrade",
                "timestamp": current.get("time", "")
            }
    except Exception as e:
        print(f"Belgrade AQI fallback API error: {e}")
    
    return {"error": "Unable to fetch AQI data"}

def get_belgrade_air_quality_history() -> list:
    """Get historical air quality data for Belgrade"""
    try:
        # Get hourly data for last 6 hours
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=44.7866&longitude=20.4489&hourly=pm10,pm2_5,temperature_2m&timezone=Europe/Belgrade&forecast_days=1&past_days=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            pm25 = hourly.get("pm2_5", [])
            pm10 = hourly.get("pm10", [])
            temp = hourly.get("temperature_2m", [])
            
            # Get last 6 hours
            result = []
            now = dt.datetime.now()
            for i in range(len(times) - 1, max(0, len(times) - 7), -1):
                if i < len(times):
                    time_str = times[i]
                    try:
                        time_obj = dt.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        if (now - time_obj.replace(tzinfo=None)).total_seconds() <= 21600:  # 6 hours
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
        print(f"Belgrade air quality history error: {e}")
    
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
            position: fixed;
            width: 100%;
            height: 100%;
        }
        
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
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
            display: grid;
            grid-template-columns: 92px minmax(0, 1.2fr) minmax(0, 0.8fr);
            grid-auto-rows: min-content;
            gap: 18px;
            align-items: start;
            height: auto;
            min-height: 100vh;
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px 20px 40px;
            width: 100%;
        }

        .top-status {
            grid-column: 2 / -1;
            background: var(--card-glass);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 16px;
            z-index: 10;
            backdrop-filter: blur(10px);
        }

        .side-nav {
            grid-column: 1;
            grid-row: 1 / -1;
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 12px 10px;
            background: var(--card-glass);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 16px;
            align-self: start;
        }

        .nav-brand {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--muted);
            text-align: center;
            padding: 8px 4px 2px;
        }

        .nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            padding: 12px 8px;
            border-radius: 14px;
            border: 1px solid transparent;
            color: var(--ink);
            text-decoration: none;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            transition: background 0.2s ease, border-color 0.2s ease;
        }

        .nav-item svg {
            width: 24px;
            height: 24px;
        }

        .nav-item.active {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent-strong);
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
        .mode-pill.camera-pill { background: rgba(99, 122, 255, 0.12); color: #3f5fb7; }

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
            grid-column: 2;
            grid-row: 2 / span 4;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow: visible;
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
            background: #f0f4f1;
            border-color: var(--border);
            color: var(--ink);
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

        .insight-block.insight-weather { grid-column: 3; grid-row: 2; }
        .insight-block.insight-history { grid-column: 3; grid-row: 3; }
        .insight-block.insight-outside { grid-column: 3; grid-row: 4; }
        .insight-block.insight-schedule { grid-column: 3; grid-row: 5; }

        .camera-shell {
            grid-column: 2 / -1;
            grid-row: 2;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 28px;
            box-shadow: var(--shadow);
        }

        .camera-shell h2 {
            font-size: 22px;
            margin-bottom: 8px;
        }

        .camera-shell p {
            color: var(--muted);
            font-size: 14px;
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
            .app-container {
                display: flex;
                flex-direction: column;
            }

            .top-status {
                position: static;
            }

            .side-nav {
                position: static;
                flex-direction: row;
                justify-content: space-between;
                align-items: center;
                width: 100%;
                padding: 10px 12px;
            }

            .nav-brand {
                text-align: left;
                padding: 0;
            }

            .nav-item {
                flex-direction: row;
                gap: 8px;
                padding: 8px 10px;
                font-size: 12px;
            }

            .main-layout {
                order: 2;
            }

            .insight-block {
                order: 3;
            }

            .camera-shell {
                order: 2;
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
        <nav class="side-nav">
            <div class="nav-brand">Mi Control</div>
            <a class="nav-item {% if page == 'purifier' %}active{% endif %}" href="/">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                    <circle cx="12" cy="12" r="8"></circle>
                    <path d="M12 4v2M12 18v2M4 12h2M18 12h2"></path>
                </svg>
                <span>Purifier</span>
            </a>
            <a class="nav-item {% if page == 'camera' %}active{% endif %}" href="/camera">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                    <rect x="3" y="7" width="14" height="10" rx="2"></rect>
                    <path d="M17 9l4-2v10l-4-2"></path>
                </svg>
                <span>Camera</span>
            </a>
        </nav>
        <!-- Top Status Bar -->
        <div class="top-status">
            <div class="status-meta">
                {% if page == 'purifier' %}
                <div class="mode-pill" id="mode-pill">Mode: --</div>
                <div class="mode-caption">Live room readings</div>
                {% else %}
                <div class="mode-pill camera-pill">Camera</div>
                <div class="mode-caption">Camera controls coming soon.</div>
                {% endif %}
                <div class="theme-switch">
                    <span class="theme-label">Theme</span>
                    <button class="theme-toggle" id="theme-toggle" type="button" onclick="toggleTheme()">Light</button>
                </div>
            </div>
            {% if page == 'purifier' %}
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
            {% endif %}
        </div>
        
        {% if page == 'purifier' %}
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
        
        <!-- Belgrade Outside Air Quality -->
        <div class="graph-section insight-block insight-outside">
            <div class="collapsible-header collapsed" onclick="toggleCollapse(this, 'belgrade-air')">
                <div class="collapsible-title">Belgrade Outside Air Quality</div>
                <div class="collapse-icon">></div>
            </div>
            <div class="collapsible-content collapsed" id="belgrade-air-content-wrapper">
                <div style="margin-bottom: 15px; padding: 12px; background: var(--card); border-radius: 12px; border: 1px solid var(--border);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 11px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.08em;">Current AQI</div>
                            <div style="font-size: 24px; font-weight: 700;" id="belgrade-aqi">--</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.08em;">PM2.5</div>
                            <div style="font-size: 16px;" id="belgrade-pm25">--</div>
                        </div>
                    </div>
                </div>
                <div class="graph-title">Outside AQI</div>
                <div class="graph-container">
                    <canvas id="belgradeAQIChart"></canvas>
                </div>
                <div class="graph-title">Outside Temperature</div>
                <div class="graph-container">
                    <canvas id="belgradeTempChart"></canvas>
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
        {% else %}
        <div class="camera-shell">
            <h2>Camera</h2>
            <p>This page is ready for the camera feed and controls. Wire up the stream and actions when you're ready.</p>
        </div>
        {% endif %}
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
        let belgradeAQIChart = null;
        let belgradeTempChart = null;
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
            loadBelgradeAirQuality();
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
        loadBelgradeAQI();
        setInterval(loadBelgradeAQI, 300000); // Update Belgrade AQI every 5 minutes
        setTimeout(() => {
            loadHistorical();
            setInterval(loadHistorical, 60000); // Update historical every minute
            loadBelgradeAirQuality();
            setInterval(loadBelgradeAirQuality, 300000); // Update Belgrade air quality every 5 minutes
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

        function loadBelgradeAQI() {
            fetch('/api/belgrade-aqi')
                .then(r => r.json())
                .then(data => {
                    if (data && !data.error) {
                        document.getElementById('belgrade-aqi').textContent = data.aqi !== null ? data.aqi : '--';
                        const pmEl = document.getElementById('belgrade-pm25');
                        if (pmEl) {
                            pmEl.innerHTML = data.pm25 !== null ? data.pm25.toFixed(1) + ' &micro;g/m&sup3;' : '--';
                        }
                    }
                })
                .catch(err => console.error('Belgrade AQI failed:', err));
        }

        function loadBelgradeAirQuality() {
            fetch('/api/belgrade-air-quality')
                .then(r => r.json())
                .then(data => {
                    if (!data.data || data.data.length === 0) return;

                    const labels = data.data.map(d => {
                        const date = new Date(d.timestamp);
                        return date.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                    });

                    const aqiData = data.data.map(d => d.aqi || 0);
                    const tempData = data.data.map(d => d.temperature || 0);

                    // Belgrade AQI Chart
                    const ctxAQI = document.getElementById('belgradeAQIChart');
                    if (ctxAQI) {
                        if (belgradeAQIChart) belgradeAQIChart.destroy();
                        belgradeAQIChart = new Chart(ctxAQI, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Belgrade AQI',
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

                    // Belgrade Temperature Chart
                    const ctxTemp = document.getElementById('belgradeTempChart');
                    if (ctxTemp) {
                        if (belgradeTempChart) belgradeTempChart.destroy();
                        belgradeTempChart = new Chart(ctxTemp, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Outside Temperature (C)',
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
                .catch(err => console.error('Belgrade air quality failed:', err));
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


@app.get("/camera")
def camera():
    """Camera dashboard placeholder"""
    snap = snapshot_state()
    return render_template_string(HTML_TEMPLATE, state=snap, page="camera")


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

@app.get("/api/belgrade-aqi")
def api_belgrade_aqi():
    """Get Belgrade AQI"""
    aqi_data = get_belgrade_aqi()
    return jsonify(aqi_data)

@app.get("/api/belgrade-air-quality")
def api_belgrade_air_quality():
    """Get Belgrade air quality history"""
    history = get_belgrade_air_quality_history()
    return jsonify({"data": history})

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


# Start control loop in background
threading.Thread(target=control_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

