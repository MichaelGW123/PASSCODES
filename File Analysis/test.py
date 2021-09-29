# Calculating percent of file found in other file

# Importing Libraries

from matplotlib import pyplot as plt
import math
import numpy as np
from pathlib import Path
import time

tester_flag = False
show_flag = True

target_file_name = 'words45to50'

generated_same_file_name = f'PRED{target_file_name}-1000'

start = time.time()
generated_file_path = Path(__file__).with_name(generated_same_file_name+'.txt')
generated_length = 0
with open(generated_file_path, 'r') as generated_file:
    for line in generated_file:
        generated_length+=1
end = time.time()
print(f'Time to read through data: {end-start}')

generated_file.close()
print(f'Generated File Length: {generated_length}')
generated_list_length_array = np.arange(generated_length+1)
print('Done')