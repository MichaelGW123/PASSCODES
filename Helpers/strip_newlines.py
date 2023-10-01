from pathlib import Path
model_ver = 5
specificFile = 'entropy_bin_00_output'
path = f'/home/morpheus/PASSCODES/Generated_Files/{model_ver}_Hidden_Layers/PRED{specificFile}-1000(RNN).txt'
path2 = f'/home/morpheus/PASSCODES/Generated_Files/{model_ver}_Hidden_Layers/PRED{specificFile}-1000(RNN)_cleared.txt'

with open(path) as infile, open(path2, 'w') as outfile:
    for line in infile:
        if not line.strip():
            continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output
