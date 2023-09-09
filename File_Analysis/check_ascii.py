import csv
import string

# Define the path to your CSV file
csv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/6sig_hashesorg2019.csv"

# Function to check if a string contains only printable ASCII characters (excluding space)


def is_printable_ascii(s):
    printable_ascii = set(string.printable) - set(" ")  # Exclude space
    return all(c in printable_ascii for c in s)


# Open the CSV file for reading
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    # Iterate through the rows and check the first column
    for row in csv_reader:
        if len(row) > 0:
            first_column_value = row[0]
            if not is_printable_ascii(first_column_value):
                print(f"Invalid: {first_column_value}")
