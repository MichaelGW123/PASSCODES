# Michael Williamson
# Masters Research
# Calculating number of passwords below and above target set (Version 1 - not NumPy)

# Importing Libraries
from pathlib import Path
import numpy as np
import time
import re

# Generated File details
target_file_name = 'words20to25'
upper_bound = 25  # exclusive
lower_bound = 20  # inclusive
generated_file_name = f'PRED{target_file_name}-1000'
generating_model = 'Markov'

less_than = []
greater_than = []

check_entropy_flag = False  # If true, checks actual entropy


def entropycheck(password):
    length = len(password)
    pool = 0
    # Finding if the password contains lowecase letters.
    lower_match = re.compile(r'[a-z]').findall(password)
    # Finding if the password contains uppercase letters.
    upper_match = re.compile(r'[A-Z]').findall(password)
    # Finding if the password contains numbers.
    number_match = re.compile(r'[0-9]').findall(password)
    # Finding if the password contains special characters.
    symbol_match = re.compile(
        r'[!"#$%&\'\(\)\*\+\,-./\:\;<>=?@\{\}\[\]\"^_`~|]').findall(password)
    if (len(lower_match) != 0):
        pool += 26
    if (len(upper_match) != 0):
        pool += 26
    if (len(number_match) != 0):
        pool += 10
    if (len(symbol_match) != 0):
        pool += 32
    if pool == 0:
        return 100000
    else:
        x = np.array([pool**length])
        entropy = np.log2(x)
    return entropy[0]


# Loops through Generated File
start = time.time()
generated_file_path = Path(__file__).with_name(generated_file_name+'.txt')
generated_length = 0
with open(generated_file_path, 'r') as generated_file:
    for line in generated_file:
        current_line = line.strip()
        generated_length += 1
        if check_entropy_flag:
            if (entropycheck(current_line) < lower_bound):
                less_than.append(current_line)
            elif (entropycheck(current_line) >= upper_bound):
                greater_than.append(current_line)
        else:
            if len(current_line) < 3:
                less_than.append(current_line)
            if len(current_line) > 7:
                greater_than.append(current_line)
                print(current_line)
end = time.time()
print(f'Time to read and process data: {end-start}')
print(f'{len(less_than)} words with entropy less than {lower_bound}')
print(f'{len(greater_than)} words with entropy greater than {upper_bound}')
