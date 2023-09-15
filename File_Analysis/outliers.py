import csv

# Define the path to your input TSV file
tsv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/hashesorg2019.tsv"

# Define the threshold (6 times the standard deviation)
threshold = 600
kept = 0
excluded = 0
lowest_pass = " "
highest_pass = " "
highest_kept = 0
lowest_outlier = 10000

# Initialize counters for processed rows
processed_rows = 0
batch_size = 10000000  # Process this many rows before printing the progress

# Define file paths for above and below threshold data
above_threshold_tsv = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/outliers_hashesorg2019.tsv"
below_threshold_tsv = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/6sig_hashesorg2019.tsv"

# Open TSV files for writing with tabs as the delimiter
with open(above_threshold_tsv, 'w', newline='') as above_tsv, \
        open(below_threshold_tsv, 'w', newline='') as below_tsv:

    above_writer = csv.writer(above_tsv, delimiter='\t')
    below_writer = csv.writer(below_tsv, delimiter='\t')

    # Open the input TSV file for reading with tabs as the delimiter
    with open(tsv_file, 'r', newline='') as tsv_input:
        tsv_reader = csv.reader(tsv_input, delimiter='\t')

        # Process the data line by line
        for row in tsv_reader:
            password = row[0]
            number = float(row[1])  # Convert the number to a float if needed

            # Check if the number is above the threshold
            if number > threshold:
                above_writer.writerow([password, number])
                kept += 1
                if (number < lowest_outlier):
                    lowest_outlier = number
                    lowest_pass = password
            else:
                below_writer.writerow([password, number])
                excluded += 1
                if (number > highest_kept):
                    highest_kept = number
                    highest_pass = password

            # Increment the processed row counter
            processed_rows += 1

            # Print progress after processing a batch of rows
            if processed_rows % batch_size == 0:
                print(f"Processed {processed_rows} rows.")

# Print the total number of rows processed
print(
    f"Total processed rows: {processed_rows}. {kept}: Highest Kept {highest_pass} - {highest_kept}. {excluded}: Lowest Outlier {lowest_pass} - {lowest_outlier}")
