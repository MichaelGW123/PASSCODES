from pathlib import Path

path = Path(__file__).parent / 'PREDwords (NC).txt'
path2 = Path(__file__).parent / 'PREDwords (Clear).txt'

with open(path) as infile, open(path2, 'w') as outfile:
    for line in infile:
        if not line.strip(): continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output