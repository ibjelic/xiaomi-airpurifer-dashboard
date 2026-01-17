# Xiaomi Air Purifier Dashboard

A beautiful web dashboard for controlling your Xiaomi Air Purifier (zhimi.airp.rmb1) with custom curve support.

<img width="918" height="450" alt="image" src="https://github.com/user-attachments/assets/c29690ad-0702-404e-816b-f303909ea774" />
<img width="550" height="439" alt="image" src="https://github.com/user-attachments/assets/baa17978-0807-494a-9bc0-fb9126194142" />
<img width="364" height="729" alt="image" src="https://github.com/user-attachments/assets/5093a0d3-1e04-4f3e-a50d-df8fe24f587b" /> <img width="353" height="705" alt="image" src="https://github.com/user-attachments/assets/3a2dcbcf-bbe4-4c31-a8e7-31c8ade9fb34" />


## Features

- **Three Control Modes:**
  - **Auto**: Device handles AQI automatically
  - **Manual**: Set a fixed fan speed percentage
  - **Favorite**: Use custom curve to map AQI to fan speed

- **Interactive Curve Editor:**
  - Click on the graph to add points
  - Right-click points to delete them
  - Visual representation of AQI vs Fan Speed relationship
  - Real-time curve interpolation

- **Real-time Status:**
  - Current AQI reading
  - Current mode
  - Target fan speed
  - Applied level
  - Last update time

## Setup

**IMPORTANT: Python Version Requirement**
- This app requires **Python 3.9, 3.10, 3.11, or 3.12**
- Python 3.13+ has compatibility issues with python-miio

### Automated Setup (Recommended - Linux)

Use the provided setup script for automatic installation and systemd service setup:

```bash
# Make script executable (first time only)
chmod +x setup.sh

# Full installation (creates venv, installs requirements, sets up service)
./setup.sh install

# Check current status
./setup.sh status

# Start the service
./setup.sh start
```

#### Setup Script Commands

| Command | Description |
|---------|-------------|
| `./setup.sh install` | Full installation (venv + requirements + service) |
| `./setup.sh refresh` | Force recreate venv and reinstall requirements |
| `./setup.sh service` | Setup/refresh systemd service only |
| `./setup.sh start` | Start the service |
| `./setup.sh stop` | Stop the service |
| `./setup.sh restart` | Restart the service |
| `./setup.sh status` | Show current status of everything |
| `./setup.sh uninstall` | Remove service (keeps venv) |
| `./setup.sh help` | Show help message |

**Options:**
- `--force` - Force recreate even if valid (e.g., `./setup.sh install --force`)

**View logs:**
```bash
journalctl -u xiaomi-airpurifier-dashboard -f
```

### Manual Setup

1. **Create a virtual environment (if not already done):**
   ```bash
   python3.12 -m venv .venv
   # or use python3.11, python3.10, python3.9
   ```

2. **Activate the virtual environment:**
   ```bash
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   # Windows CMD
   .venv\Scripts\activate.bat

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your device:**
   - Copy `.env.example` to `.env`
   - Fill in your device IP and token:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your values:
     - `MIIO_IP`: Your air purifier's local IP address
     - `MIIO_TOKEN`: Your device token (get from Mi Home app or use `miio discover`)

5. **Get your device token:**
   - Use the Mi Home app to get the token
   - Or use `miio discover` command (if you have python-miio installed)
   - Or use tools like `miio-extract-tokens` from Home Assistant

6. **Run the application:**
   ```bash
   python app.py
   ```

7. **Access the dashboard:**
   - Open your browser to `http://localhost:5000`
   - The dashboard will automatically update every 5 seconds

## Usage

### Setting Modes

Click on one of the three mode buttons:
- **Auto**: Let the device automatically adjust based on AQI
- **Manual**: Set a fixed percentage using the slider
- **Favorite**: Use your custom curve to control fan speed based on AQI

### Creating a Custom Curve

1. Select **Favorite** mode
2. Click anywhere on the graph to add a point
3. Points are automatically sorted by AQI value
4. The curve will interpolate between points
5. Right-click a point to delete it (minimum 2 points required)
6. Click **Save Curve** to apply your settings

### Example Curve

A typical curve might look like:
- AQI 0 → 10% fan speed (quiet)
- AQI 50 → 30% fan speed (low)
- AQI 100 → 50% fan speed (medium)
- AQI 150 → 70% fan speed (high)
- AQI 300 → 100% fan speed (maximum)

## Configuration

Edit `.env` to customize:

- `LEVEL_MIN` / `LEVEL_MAX`: Fan level range (default: 1-14)
- `POLL_INTERVAL`: How often to check status (default: 5 seconds)
- `PORT`: Web server port (default: 5000)

## Troubleshooting

- **Connection errors**: Make sure your device IP and token are correct
- **No AQI reading**: Check that your device is online and connected to the same network
- **Mode not changing**: Verify your device supports the requested mode

## Requirements

- **Python 3.9, 3.10, 3.11, or 3.12** (Python 3.13+ is NOT compatible due to pydantic v1 issues)
- Flask 3.0.3
- python-miio (latest from git)
- python-dotenv 1.0.0

## Notes

- The dashboard runs a background control loop that continuously monitors and updates the device
- In Favorite mode, the curve is used to automatically adjust fan speed based on current AQI
- The device must be on the same local network as the server

