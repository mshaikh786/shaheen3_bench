#!/bin/bash

# Define the log file
LOG_FILE="system_insights.log"

# Create or clear the log file
> "$LOG_FILE"

# Function to log a section with a header
log_section() {
  echo "========================================" >> "$LOG_FILE"
  echo "$1" >> "$LOG_FILE"
  echo "========================================" >> "$LOG_FILE"
  echo >> "$LOG_FILE"
}

# Collect and log system insights

log_section "Operating System Information"
cat /etc/os-release >> "$LOG_FILE" 2>&1

log_section "Kernel Version"
uname -r >> "$LOG_FILE" 2>&1

log_section "CPU Information"
lscpu >> "$LOG_FILE" 2>&1

log_section "Memory Information (in MB)"
free -m >> "$LOG_FILE" 2>&1

log_section "OFED Information"
ofed_info -n >> "$LOG_FILE" 2>&1

log_section "InfiniBand Status"
ibstat >> "$LOG_FILE" 2>&1

log_section "Disk Information"
lsblk >> "$LOG_FILE" 2>&1

log_section "NVIDIA GPU Summary"
nvidia-smi >> "$LOG_FILE" 2>&1

log_section "Detailed NVIDIA GPU Information"
nvidia-smi -a >> "$LOG_FILE" 2>&1

# Print completion message
echo "System insights have been logged to $LOG_FILE"

