from pathlib import Path
specificFile = 'words80to90'
path = Path(__file__).parent / f'PRED{specificFile}-1000(RNN).txt'
path2 = Path(__file__).parent / f'PRED{specificFile}-1000(RNN)(Clear).txt'

with open(path) as infile, open(path2, 'w') as outfile:
    for line in infile:
        if not line.strip():
            continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output
