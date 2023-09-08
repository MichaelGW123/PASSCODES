import string

# Define a set of printable ASCII characters
printable_ascii_characters = set(string.printable[:94])


def strip_non_printable_ascii(line):
    # Filter out non-printable ASCII characters
    stripped_line = ''.join(
        char for char in line if char in printable_ascii_characters)
    return stripped_line


def main():
    input_file_path = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/wordslessthan20.txt"
    output_file_path = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/stripped_wordslessthan20.txt"

    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            line_number = 1
            for line in input_file:
                stripped_line = strip_non_printable_ascii(line)
                # Write the stripped line to the output file
                output_file.write(stripped_line + '\n')
                line_number += 1
        print(
            f"File checked and stripped lines written to '{output_file_path}' successfully")
    except FileNotFoundError:
        print(f"Input file '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
