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
main_directories=("Generated_Files" "model_weights" "training_checkpoints" "Training_Graphs")
subdirectories=("1 Hidden Layers" "2 Hidden Layers" "3 Hidden Layers")

for main_dir in "${main_directories[@]}"; do
    create_directory "$main_dir"

    if [[ "$main_dir" == "Source_Files" ]]; then
        continue  # Skip subdirectories for "Source_Files"
    fi

    cd "$main_dir" || exit

    for subdir in "${subdirectories[@]}"; do
        create_directory "$subdir"
    done

    cd .. || exit
done
