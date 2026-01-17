#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="xiaomi-airpurifier-dashboard"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
APP_FILE="${SCRIPT_DIR}/app.py"
VALID_PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Find a valid Python interpreter
find_python() {
    local python_cmd=""

    # Try python3.12, python3.11, python3.10, python3.9 in order
    for version in "3.12" "3.11" "3.10" "3.9"; do
        if command -v "python${version}" &> /dev/null; then
            python_cmd="python${version}"
            break
        fi
    done

    # Fallback to python3
    if [ -z "$python_cmd" ] && command -v python3 &> /dev/null; then
        python_cmd="python3"
    fi

    # Fallback to python
    if [ -z "$python_cmd" ] && command -v python &> /dev/null; then
        python_cmd="python"
    fi

    echo "$python_cmd"
}

# Validate Python version
validate_python_version() {
    local python_cmd="$1"
    local version_output
    local major_minor

    if [ -z "$python_cmd" ]; then
        print_error "No Python interpreter found!"
        return 1
    fi

    version_output=$($python_cmd --version 2>&1)
    major_minor=$(echo "$version_output" | grep -oP '\d+\.\d+' | head -1)

    for valid_version in "${VALID_PYTHON_VERSIONS[@]}"; do
        if [ "$major_minor" == "$valid_version" ]; then
            print_success "Found valid Python version: $version_output"
            return 0
        fi
    done

    print_error "Invalid Python version: $version_output"
    print_error "Required: Python 3.9, 3.10, 3.11, or 3.12"
    return 1
}

# Check if venv is valid
check_venv_valid() {
    if [ ! -d "$VENV_DIR" ]; then
        return 1
    fi

    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        return 1
    fi

    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
        return 1
    fi

    # Check if venv Python version is valid
    local venv_version
    venv_version=$("${VENV_DIR}/bin/python" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)

    for valid_version in "${VALID_PYTHON_VERSIONS[@]}"; do
        if [ "$venv_version" == "$valid_version" ]; then
            return 0
        fi
    done

    return 1
}

# Check if requirements are installed
check_requirements_installed() {
    if [ ! -f "${VENV_DIR}/bin/pip" ]; then
        return 1
    fi

    # Check if key packages are installed
    if ! "${VENV_DIR}/bin/python" -c "import flask; import dotenv; import miio" &> /dev/null; then
        return 1
    fi

    return 0
}

# Setup virtual environment
setup_venv() {
    local python_cmd="$1"
    local force_recreate="$2"

    if [ "$force_recreate" == "true" ] && [ -d "$VENV_DIR" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    if check_venv_valid; then
        print_success "Virtual environment already exists and is valid"
        return 0
    fi

    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment exists but is invalid, recreating..."
        rm -rf "$VENV_DIR"
    fi

    print_info "Creating virtual environment with $python_cmd..."
    $python_cmd -m venv "$VENV_DIR"

    if check_venv_valid; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create valid virtual environment"
        return 1
    fi
}

# Install requirements
install_requirements() {
    local force_reinstall="$1"

    if [ "$force_reinstall" != "true" ] && check_requirements_installed; then
        print_success "Requirements already installed"
        return 0
    fi

    print_info "Upgrading pip..."
    "${VENV_DIR}/bin/pip" install --upgrade pip

    print_info "Installing requirements..."
    "${VENV_DIR}/bin/pip" install -r "$REQUIREMENTS_FILE"

    if check_requirements_installed; then
        print_success "Requirements installed successfully"
    else
        print_error "Failed to install requirements"
        return 1
    fi
}

# Check if service exists and is correct
check_service_valid() {
    if [ ! -f "$SERVICE_FILE" ]; then
        return 1
    fi

    # Check if service file points to correct paths
    if ! grep -q "$SCRIPT_DIR" "$SERVICE_FILE" 2>/dev/null; then
        return 1
    fi

    if ! grep -q "${VENV_DIR}/bin/python" "$SERVICE_FILE" 2>/dev/null; then
        return 1
    fi

    return 0
}

# Get current user
get_service_user() {
    if [ "$EUID" -eq 0 ]; then
        # Running as root, use SUDO_USER if available
        if [ -n "$SUDO_USER" ]; then
            echo "$SUDO_USER"
        else
            echo "root"
        fi
    else
        echo "$USER"
    fi
}

# Setup systemd service
setup_service() {
    local force_recreate="$1"
    local service_user
    service_user=$(get_service_user)

    if [ "$force_recreate" != "true" ] && check_service_valid; then
        print_success "Service already exists and is valid"
        return 0
    fi

    print_info "Creating systemd service..."

    local service_content="[Unit]
Description=Xiaomi Air Purifier Dashboard
After=network.target

[Service]
Type=simple
User=${service_user}
WorkingDirectory=${SCRIPT_DIR}
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${VENV_DIR}/bin/python ${APP_FILE}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"

    if [ "$EUID" -ne 0 ]; then
        print_warning "Need sudo to create service file"
        echo "$service_content" | sudo tee "$SERVICE_FILE" > /dev/null
    else
        echo "$service_content" > "$SERVICE_FILE"
    fi

    print_info "Reloading systemd daemon..."
    if [ "$EUID" -ne 0 ]; then
        sudo systemctl daemon-reload
    else
        systemctl daemon-reload
    fi

    print_success "Service created successfully"
}

# Enable and start service
enable_service() {
    print_info "Enabling service..."
    if [ "$EUID" -ne 0 ]; then
        sudo systemctl enable "$SERVICE_NAME"
    else
        systemctl enable "$SERVICE_NAME"
    fi
    print_success "Service enabled"
}

start_service() {
    print_info "Starting service..."
    if [ "$EUID" -ne 0 ]; then
        sudo systemctl start "$SERVICE_NAME"
    else
        systemctl start "$SERVICE_NAME"
    fi
    print_success "Service started"
}

restart_service() {
    print_info "Restarting service..."
    if [ "$EUID" -ne 0 ]; then
        sudo systemctl restart "$SERVICE_NAME"
    else
        systemctl restart "$SERVICE_NAME"
    fi
    print_success "Service restarted"
}

stop_service() {
    print_info "Stopping service..."
    if [ "$EUID" -ne 0 ]; then
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    else
        systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    fi
    print_success "Service stopped"
}

# Show service status
show_status() {
    echo ""
    print_info "=== Current Status ==="
    echo ""

    # Python
    local python_cmd
    python_cmd=$(find_python)
    if [ -n "$python_cmd" ]; then
        local version
        version=$($python_cmd --version 2>&1)
        print_info "System Python: $version ($python_cmd)"
    else
        print_warning "System Python: Not found"
    fi

    # Venv
    if check_venv_valid; then
        local venv_version
        venv_version=$("${VENV_DIR}/bin/python" --version 2>&1)
        print_success "Virtual Environment: Valid ($venv_version)"
    elif [ -d "$VENV_DIR" ]; then
        print_warning "Virtual Environment: Exists but invalid"
    else
        print_warning "Virtual Environment: Not created"
    fi

    # Requirements
    if check_requirements_installed; then
        print_success "Requirements: Installed"
    else
        print_warning "Requirements: Not installed or incomplete"
    fi

    # Service
    if [ -f "$SERVICE_FILE" ]; then
        if check_service_valid; then
            print_success "Service File: Valid"
        else
            print_warning "Service File: Exists but may be outdated"
        fi

        if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
            print_success "Service Status: Running"
        else
            print_warning "Service Status: Not running"
        fi

        if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
            print_success "Service Enabled: Yes"
        else
            print_warning "Service Enabled: No"
        fi
    else
        print_warning "Service: Not installed"
    fi

    echo ""
}

# Show help
show_help() {
    echo "Xiaomi Air Purifier Dashboard Setup Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  install       Full installation (venv + requirements + service)"
    echo "  refresh       Refresh/recreate venv and reinstall requirements"
    echo "  service       Setup/refresh service only"
    echo "  start         Start the service"
    echo "  stop          Stop the service"
    echo "  restart       Restart the service"
    echo "  status        Show current status"
    echo "  uninstall     Remove service (keeps venv)"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  --force       Force recreate even if valid"
    echo ""
    echo "Examples:"
    echo "  $0 install          # Full installation"
    echo "  $0 refresh          # Refresh venv and requirements"
    echo "  $0 status           # Check current status"
    echo "  $0 install --force  # Force full reinstall"
}

# Uninstall service
uninstall_service() {
    print_info "Uninstalling service..."

    if [ "$EUID" -ne 0 ]; then
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        sudo rm -f "$SERVICE_FILE"
        sudo systemctl daemon-reload
    else
        systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        rm -f "$SERVICE_FILE"
        systemctl daemon-reload
    fi

    print_success "Service uninstalled"
}

# Main installation
do_install() {
    local force="$1"

    echo ""
    print_info "=== Xiaomi Air Purifier Dashboard Setup ==="
    echo ""

    # Find and validate Python
    local python_cmd
    python_cmd=$(find_python)

    if ! validate_python_version "$python_cmd"; then
        exit 1
    fi

    # Setup venv
    setup_venv "$python_cmd" "$force"

    # Install requirements
    install_requirements "$force"

    # Setup service
    setup_service "$force"

    # Enable service
    enable_service

    echo ""
    print_success "=== Installation Complete ==="
    echo ""
    print_info "To start the service: $0 start"
    print_info "To check status: $0 status"
    print_info "To view logs: journalctl -u $SERVICE_NAME -f"
    echo ""
}

# Refresh installation
do_refresh() {
    echo ""
    print_info "=== Refreshing Installation ==="
    echo ""

    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        stop_service
    fi

    # Find and validate Python
    local python_cmd
    python_cmd=$(find_python)

    if ! validate_python_version "$python_cmd"; then
        exit 1
    fi

    # Recreate venv
    setup_venv "$python_cmd" "true"

    # Reinstall requirements
    install_requirements "true"

    # Refresh service
    setup_service "true"

    print_success "=== Refresh Complete ==="
    echo ""
    print_info "To start the service: $0 start"
    echo ""
}

# Parse arguments
COMMAND="${1:-install}"
FORCE="false"

for arg in "$@"; do
    case $arg in
        --force)
            FORCE="true"
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    install)
        do_install "$FORCE"
        ;;
    refresh)
        do_refresh
        ;;
    service)
        setup_service "$FORCE"
        enable_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    uninstall)
        uninstall_service
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
