#!/bin/bash

# Function to create a directory
create_directory() {
    if mkdir "$1"; then
        echo "Directory '$1' created successfully."
    else
        echo "Error creating directory '$1'."
    fi
}

# Main script
main_directories=("Generated_Files" "model_weights" "Source_Files" "training_checkpoints" "Training_Graphs")

for main_dir in "${main_directories[@]}"; do
    create_directory "$main_dir"

    if [[ "$main_dir" == "Source_Files" ]]; then
        create_directory "entropy_data"
        create_directory "text_data"
        continue 
    fi
    if [[ "$main_dir" == "training_checkpoints" ]]; then
        continue 
    fi

    cd .. || exit
done
