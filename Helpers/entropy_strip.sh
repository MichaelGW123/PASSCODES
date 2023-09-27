#!/bin/bash

# Input directory containing .tsv files and output directory for .txt files
input_dir="/home/morpheus/PASSCODES/Source_Files/entropy_data"

# Column separator (assuming tab-separated values)
separator=$'\t'

# Define a function to process a single .tsv file
process_tsv_file() {
    local tsv_file="$1"
    local base_name
    base_name="$(basename "$tsv_file" .tsv)"
    output_dir="/home/morpheus/PASSCODES/Source_Files/text_data"
    local output_file="$output_dir/${base_name}.txt"
    > "$output_file"
    cut -f 1 "$tsv_file" > "$output_file"
}

# Export the function so it's available to parallel processes
export -f process_tsv_file

# Use find and parallel to process .tsv files in parallel
find "$input_dir" -type f -name "*.tsv" -print0 | parallel -0 process_tsv_file {}
