import csv


def main():
    input_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/csv_wordslessthan20.csv"
    output_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/sorted_csv_wordslessthan20.csv"

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        header = next(csv_reader)  # Read the header

        # Sort the data by entropy (second column) numerically
        sorted_data = sorted(csv_reader, key=lambda row: float(row[1]))

        csv_writer = csv.writer(outfile)
        csv_writer.writerow(header)  # Write the header
        csv_writer.writerows(sorted_data)

    print(f"Password entropy data sorted and saved to {output_file}")


if __name__ == "__main__":
    main()
