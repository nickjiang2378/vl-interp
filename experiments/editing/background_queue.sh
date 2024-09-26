#!/bin/bash

# File to store the command queue
QUEUE_FILE="./command_queue.txt"
# File to store executed commands
LOG_FILE="./command_log.txt"

# Function to add a command to the queue
add_to_queue() {
    echo "$1" >> "$QUEUE_FILE"
    echo "Command added to queue: $1"
}

# Function to log executed command
log_command() {
    local cmd="$1"
    local status="$2"
    local execution_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$execution_time] [Status: $status] $cmd" >> "$LOG_FILE"
}

# Function to process the queue
process_queue() {
    local gpu_device="$1"

    while true; do
        # Read the first line of the file
        read -r cmd < "$QUEUE_FILE"

        # Exit the loop if the file is empty
        if [ -z "$cmd" ]; then
            break
        fi

        echo "Executing on GPU $gpu_device: $cmd"

        # Remove the first line from the file
        sed -i '1d' "$QUEUE_FILE"

        # Execute the command with CUDA_VISIBLE_DEVICES set
        if eval "CUDA_VISIBLE_DEVICES=$gpu_device $cmd"; then
            echo "Command completed successfully: $cmd"
            # Log the successful command
            log_command "$cmd" "SUCCESS"
        else
            echo "Command failed: $cmd"
            # Log the failed command
            log_command "$cmd" "FAILED"
        fi
    done
}

# Main script logic
case "$1" in
    add)
        shift
        add_to_queue "$*"
        ;;
    process)
        if [ -z "$2" ]; then
            echo "Error: GPU device number is required for processing."
            echo "Usage: $0 process <gpu_device_number>"
            exit 1
        fi
        process_queue "$2"
        ;;
    *)
        echo "Usage: $0 {add 'command' | process <gpu_device_number>}"
        exit 1
        ;;
esac