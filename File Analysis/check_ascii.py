import string

# Define a set of printable ASCII characters
printable_ascii_characters = set(string.printable[:94])


def find_invalid_characters(line):
    # Remove newline characters and find invalid characters in the line
    return [char for char in line if char not in printable_ascii_characters]


def main():
    # Get the file name from the user
    file_name = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/stripped_wordslessthan20.txt"

    try:
        with open(file_name, 'r') as file:
            line_number = 1
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                invalid_characters = find_invalid_characters(line)
                if invalid_characters:
                    invalid_characters_str = ''.join(invalid_characters)
                    print(
                        f"Line {line_number}: Invalid characters found: {invalid_characters_str}")
                line_number += 1
        print("File checked successfully")
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
