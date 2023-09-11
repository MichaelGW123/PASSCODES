import csv

# Define the path to your input CSV file
csv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/hashesorg2019.csv"

# Define the threshold (6 times the standard deviation)
threshold = 600
lowest_pass = " "
highest_pass = " "
highest_kept = 0
lowest_outlier = 10000

# Initialize counters for processed rows
processed_rows = 0
batch_size = 10000000  # Process this many rows before printing the progress

# Define file paths for above and below threshold data
above_threshold_csv = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/outliers_hashesorg2019.csv"
below_threshold_csv = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/6sig_hashesorg2019.csv"

# Open CSV files for writing
with open(above_threshold_csv, 'w', newline='') as above_csv, \
        open(below_threshold_csv, 'w', newline='') as below_csv:

    above_writer = csv.writer(above_csv)
    below_writer = csv.writer(below_csv)

    # Open the input CSV file for reading
    with open(csv_file, 'r') as csv_input:
        csv_reader = csv.reader(csv_input)

        # Process the data line by line
        for row in csv_reader:
            password = row[0]
            number = float(row[1])  # Convert the number to a float if needed

            # Check if the number is above the threshold
            if number > threshold:
                above_writer.writerow([password, number])
                if (number < lowest_outlier):
                    lowest_outlier = number
                    lowest_pass = password
            else:
                below_writer.writerow([password, number])
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
    f"Total processed rows: {processed_rows}. Highest Kept {highest_pass} - {highest_kept} and Lowest Outlier {lowest_pass} - {lowest_outlier}")
