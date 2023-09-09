# Password Generation and Cracking Potential Comparison

## Overview

This project explores the generation of passwords using Recurrent Neural Networks (RNNs) and Markov models, with the aim of comparing their potential for password cracking. The goal is to analyze the models' ability to generate realistic passwords and effectively leverage statistically recognizable patterns in lower-entropy passwords to compromise higher-entropy passwords.

## Inspiration

(Discuss original inspiration)

## Features

- Password generation using RNN and Markov models.
- Comparison of generated passwords' complexity and randomness.
- Password cracking simulations using common techniques.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/MichaelGW123/DeepLearningEntropy.git
   cd DeepLearningEntropy
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Right now this simply is used to generate files and then we compare that, but this section will be expanded as the orignal implementation of generating the passwords is improved.

1. Using the training set, seperate it into different entropies of any desired granularity.

- This step starts off with using the 'hashesorg2019' file. File_Analysis/basic_process.py will create a csv, with each row containing [password (column 0), entropy (column 1)]. Entropy will be calculated by considering the unique pools of characters pulled from (ex. lowercase alphabetical, uppercase alphabetical, numerical, and special symbols) to assemble the character set then raised to the power of the length of the password, then the log2 computed. This will make it easier to modify later as we will not need to recalcuate the entropy.

- File_Analysis/outliers.py will generate 2 CSVs based on the input CSV. It has a hard coded threshold for mean + 6 \* std_dev for the hashesorg2019.csv file. It will separate those with entropy above that limit and those below into their respective CSV file.

- File_Analysis/kde.py (currently doesn't work because of memory constraints) will generate a KDE of the entropy for the passwords.

- File_Analysis/sort_file.py will sort a file in ascending order of entropy written to a new CSV file.

2. Determine how you want to test effectiveness. For example, you can test in the same entropy bin by using a test and training set, or aim for the entropy set above the one you train with (therefore allowing you to use 100% of the entropy bin below your target set)

3. Train your model on the desired set.

4. Generate a number of passwords with the Markov and/or RNN model.

5. See what percent of passwords were a match with you intended target set.

6. (Optional) Attempt more generic password cracking methods and compare run time as well as number of generated guesses to achieve comprable results.

## Results

TBA

## Potential Future Improvements

## Contributing

Me, Myself, and I (so far)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- This project was inspired by the need to better understand password security and the effectiveness of different generation methods.
- The code structure and some utility functions were adapted from (update with actual references).
