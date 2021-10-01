# Calculating percent of file found in other file

# Importing Libraries

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import time

tester_flag = False
show_flag = False

same_file_flag = True
target_file_name = 'words45to50'
generated_lower_file_name = 'Generated'
generated_same_file_name = f'PRED{target_file_name}-1000'

if same_file_flag:
    generated_file_name = generated_same_file_name
else:
    generated_file_name = generated_lower_file_name

generating_model = 'Markov'

start = time.time()
target_file_path = Path(__file__).with_name(target_file_name+'.txt')
target_file = open(target_file_path, 'r')
unique_target_words=set(line.strip() for line in target_file)
target_length = len(unique_target_words)
target_file.close()
end = time.time()
print(f'Time to read and create set of Target dataset: {end-start}')
print(f'Target File Unique Occurences Length: {target_length}')

array_percentages = [0]
found = 0
percent = found/target_length

generated_file_path = Path(__file__).with_name(generated_file_name+'.txt')
start = time.time()
generated_length = 0
with open(generated_file_path, 'r') as generated_file:
    for line in generated_file:
        current_line = line.strip()
        generated_length+=1
        if generated_length%50000000 == 0:
            print(f'{generated_length} completed')
        if current_line in unique_target_words:
            found += 1
            unique_target_words.remove(current_line)
            percent = (found/target_length)*100
            array_percentages.append(percent)
        else:
            array_percentages.append(percent)
array_percentages = np.array(array_percentages)
end = time.time()
print(f'Time to read and process data: {end-start}')
print(f'Generated File Length: {generated_length}')
generated_list_length_array = np.arange(generated_length+1)

if tester_flag:
    print(f'Array of Percentages: {array_percentages}')
    print(f'Array of Guesses: {generated_list_length_array}')

x = generated_list_length_array
y = array_percentages
plt.xlabel("Number of Generated Guesses")
plt.ylabel("Percent (%) Matching Target File")
if same_file_flag:
    plt.title(f'Generated Matching Effectiveness - {generating_model}\nTarget: {target_file_name} (Same Set Used to Train)')
else:
    plt.title(f'Generated Matching Effectiveness - {generating_model}\nTarget: {target_file_name}')
plt.xlim([0,generated_list_length_array[generated_length-1]])
plt.ylim([0,100])
plt.yticks(np.arange(0, 100.1, 5))
plt.xticks(np.arange(0, generated_length+1, generated_length//10))
plt.plot(x,y)
if show_flag:
    plt.show()
else:
    if same_file_flag:
        file_name = f'{target_file_name}-Same'
    else:
        file_name = f'{target_file_name}-Lower'
    plt.savefig(f'{file_name}.png')