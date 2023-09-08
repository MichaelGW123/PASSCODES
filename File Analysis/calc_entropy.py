import csv
import math


def calculate_entropy(password):
    # Calculate the entropy of a password using Shannon's entropy formula
    if len(password) == 0:
        return 0

    char_set = set(password)
    entropy = -sum((password.count(char) / len(password)) *
                   math.log2(password.count(char) / len(password)) for char in char_set)
    return entropy


def main():
    input_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/stripped_wordslessthan20.txt"
    output_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/csv_wordslessthan20.csv"

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Password', 'Entropy'])  # Write CSV header

        for line in infile:
            password = line.strip()  # Remove leading/trailing whitespace
            entropy = calculate_entropy(password)
            csv_writer.writerow([password, entropy])

    print(f"Password entropy calculated and written to {output_file}")


if __name__ == "__main__":
    main()
