# Michael Williamson
# Masters Research
# Calculating number of passwords below and above target set (Version 2 - NumPy)

# Importing Libraries
from pathlib import Path
import numpy as np
import time
import re

# Generated File details
target_file_name = 'wordslessthan20'
upper_bound = 20 #exclusive
lower_bound = 0 #inclusive
generating_model = 'Markov'
generated_file_name = f'PRED{target_file_name}-1000({generating_model})'


less_than = []
greater_than = []

password_length = []
password_pool = []

check_entropy_flag = False # If true, checks actual entropy

# Loops through Generated File
start = time.time()
generated_file_path = Path(__file__).with_name(generated_file_name+'.txt')
generated_length = 0
with open(generated_file_path, 'r') as generated_file:
    for line in generated_file:
        current_line = line.strip()
        generated_length+=1
        password_length.append(len(current_line))
        pool = 0
        lower_match = re.compile(r'[a-z]').findall(current_line)  # Finding if the password contains lowecase letters.
        upper_match = re.compile(r'[A-Z]').findall(current_line)  # Finding if the password contains uppercase letters.
        number_match = re.compile(r'[0-9]').findall(current_line)  # Finding if the password contains numbers.
        symbol_match = re.compile(r'[!"#$%&\'\(\)\*\+\,-./\:\;<>=?@\{\}\[\]\"^_`~|]').findall(current_line)  # Finding if the password contains special characters.
        if (len(lower_match) != 0):
            pool += 26
        if (len(upper_match) != 0):
            pool += 26
        if (len(number_match) != 0):
            pool += 10
        if (len(symbol_match) != 0):
            pool += 32
        password_pool.append(pool)
password_length = np.array(password_length)
password_pool = np.array(password_pool)
power = password_pool**password_length
result = np.log2(power)
for i in range(50):
    print(result[i])
end = time.time()
print(f'Time to read and process data: {end-start}')
print(f'{len(less_than)} words with entropy less than {lower_bound}')
print(f'{len(greater_than)} words with entropy greater than {upper_bound}')