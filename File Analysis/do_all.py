import string
import csv
import math
import re
import time


def calculate_entropy(password):
    password_length = len(password)
    char_set_size = 0
    lower_match = re.compile(r'[a-z]').findall(password)
    upper_match = re.compile(r'[A-Z]').findall(password)
    number_match = re.compile(r'\d').findall(password)
    symbol_match = re.compile(
        r'[!"#$%&\'\(\)\*\+\,-./\:\;<=>?@\[\\\]^_`\{|\}~]').findall(password)

    if len(lower_match) != 0:
        char_set_size += 26
    if len(upper_match) != 0:
        char_set_size += 26
    if len(number_match) != 0:
        char_set_size += 10
    if len(symbol_match) != 0:
        char_set_size += 32

    if char_set_size == 0:
        return 100000
    else:
        entropy = math.log2(char_set_size ** password_length)
    return entropy


def process_file(input_file, output_file):
    valid_lines = []

    total_lines = 0
    processed_lines = 0

    # Start the timer
    start_time = time.time()

    with open(input_file, 'rb') as file:
        for line_bytes in file:
            try:
                line = line_bytes.decode('utf-8', errors='ignore')
                line = line.strip()
                total_lines += 1

                if all(char in string.printable[1:-6] for char in line):
                    entropy = calculate_entropy(line)
                    valid_lines.append((line, entropy))

                processed_lines += 1

                if processed_lines % 10000000 == 0:
                    # Calculate the time elapsed
                    elapsed_time = time.time() - start_time
                    print(
                        f"Processed {processed_lines}/{total_lines} lines in {elapsed_time:.2f} seconds")

                    valid_lines.sort(key=lambda x: x[1])

                    with open(output_file, 'a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerows(valid_lines)

                    valid_lines = []

            except UnicodeDecodeError:
                pass

    if valid_lines:
        with open(output_file, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(valid_lines)


if __name__ == "__main__":
    input_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/hashesorg2019"
    output_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/hashesorg2019.csv"

    process_file(input_file, output_file)
