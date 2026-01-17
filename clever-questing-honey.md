# Implementation Plan: Logging, Log Explorer & Analyze Features

## Overview
Add CSV data logging, a log file explorer, and an analysis feature to generate optimized curves from historical data.

---

## 1. New Environment Variables (.env)

```bash
MIN_LOG_INTERVAL=10           # Minimum seconds between log entries
MAX_ROWS_PER_FILE=600000      # Max rows per CSV before rotation
LOG_ENABLED_DEFAULT=false     # Default logging state on startup
```

---

## 2. Log File Structure

```
/home/pyr/Desktop/xiaomi-airpurifer-dashboard/
├── app.py
├── logs/                          # Auto-created when logging enabled
│   ├── aqi/
│   │   └── aqi_20240115_143022.csv
│   ├── temperature/
│   │   └── temperature_20240115_143022.csv
│   ├── humidity/
│   │   └── humidity_20240115_143022.csv
│   ├── fan_speed/
│   │   └── fan_speed_20240115_143022.csv
│   ├── outside_aqi/
│   │   └── outside_aqi_20240115_143022.csv
│   └── outside_temp/
│       └── outside_temp_20240115_143022.csv
```

CSV format: `timestamp,value` (ISO timestamp, numeric value)

---

## 3. Backend Changes (app.py)

### 3.1 New LogManager Class (~150 lines)
- Thread-safe CSV writing with per-variable locks
- Automatic file rotation at 600k rows
- Directory creation and file naming
- Methods: `log_value()`, `get_log_tree()`, `read_logs_for_analysis()`

### 3.2 New logging_loop Thread
- Runs alongside existing control_loop
- Logs indoor values (aqi, temp, humidity, fan_speed) at configured interval
- Logs outdoor values (outside_aqi, outside_temp) every 5 minutes when available
- Respects MIN_LOG_INTERVAL from .env

### 3.3 New AnalysisEngine Class (~200 lines)
- `get_available_data_range()` - Check how much logged data exists
- `generate_24h_curve()` - Create AQI→fan_speed curve from 48h+ data
- `generate_weekly_curve()` - Create curves from 7+ days, optionally per-weekday

### 3.4 New API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/logging/status` | GET | Get logging enabled/interval |
| `/api/logging/enable` | POST | Enable/disable logging |
| `/api/logging/interval` | POST | Set logging interval |
| `/api/logs/tree` | GET | Get file tree structure |
| `/api/logs/file/<path>` | GET | Preview file content |
| `/api/logs/download/<path>` | GET | Download CSV file |
| `/api/analysis/data-range` | GET | Get available data hours |
| `/api/analysis/generate-curve` | POST | Generate curve from logs |

### 3.5 State Changes
Add to state.json: `logging_enabled`, `log_interval`

---

## 4. Frontend Changes (embedded in app.py)

### 4.1 New CSS (~150 lines)
- Modal overlay for log explorer popup
- Stylish checkboxes (custom `.checkbox-stylish` class)
- Log folder/file list styling
- Analysis section styling
- Weekday selector buttons
- Curve preview container

### 4.2 New HTML Sections

**Data Logging Section** (collapsible):
- Toggle switch for enable/disable
- Slider + number input for interval (min from .env)
- "List Logs" button to open explorer

**Analysis Section** (collapsible):
- Data range status display
- Analysis type dropdown (24h / Weekly)
- Time period selector (48h, 7d, 14d, 30d)
- Weekday selector (Mon-Sun buttons, for weekly mode)
- "Include Outside AQI" checkbox
- Max cutoff slider (20-100%)
- Curve name input
- "Generate Curve" button
- Preview chart with "Load in Curve Editor" button

**Log Explorer Modal**:
- Overlay with centered modal
- Header with close button
- Folder tree with expandable sections
- File list with size, rows, preview/download buttons

### 4.3 New JavaScript (~200 lines)
- `loadLoggingStatus()`, `updateLogging()`, `updateLogInterval()`
- `openLogExplorer()`, `closeLogExplorer()`, `loadLogTree()`
- `loadAnalysisDataRange()`, `updateAnalysisOptions()`
- `generateCurve()`, `showCurvePreview()`, `loadGeneratedCurve()`
- Modal event handlers (escape key, overlay click)

---

## 5. Implementation Order

### Phase 1: Logging Foundation
1. Add .env variables
2. Add state keys, update load_state/save_state
3. Create LogManager class
4. Implement logging_loop thread
5. Add logging API endpoints
6. Add logging UI section

### Phase 2: Log Explorer
7. Add log tree and file reading to LogManager
8. Add log explorer API endpoints
9. Create modal HTML/CSS
10. Add JavaScript for explorer

### Phase 3: Analysis
11. Create AnalysisEngine class
12. Add analysis API endpoints
13. Add analysis UI section
14. Add curve preview chart
15. Connect to curve editor

### Phase 4: Polish
16. Add stylish checkbox CSS
17. Test all features
18. Handle edge cases (no data, disk full)

---

## 6. Verification Plan

1. **Logging**: Enable logging, wait 1-2 minutes, verify CSV files created in logs/ folder
2. **Log Explorer**: Click "List Logs", verify folder tree shows, test preview and download
3. **Analysis**: After accumulating data, test curve generation with different options
4. **Curve Loading**: Generate curve, load in editor, verify points display correctly
5. **State Persistence**: Restart app, verify logging settings persist

---

## 7. Files to Modify

- `/home/pyr/Desktop/xiaomi-airpurifer-dashboard/app.py` - All backend and frontend changes
- `/home/pyr/Desktop/xiaomi-airpurifer-dashboard/.env` - Add new configuration variables
