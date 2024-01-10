# Password Analysis with Sequential Structures using Computational Openings for Defensive and Evaluative Security

## Overview

This project explores the generation of passwords using Recurrent Neural Network (RNN) models, with the aim of comparing their potential for password cracking. The goal is to analyze the models' ability to generate realistic passwords and effectively leverage statistically recognizable patterns in lower-entropy passwords to compromise higher-entropy passwords.

## Inspiration

Password cracking has been a topic of some fascination. Some of the most notable hacks were not done through brute force but rather an element of Social Engineering. Passwords hold a very unique position where they must be difficult to guess but easy to remember, which seem to be contradict each other. One method of password cracking, the dictionary attack, relies on guessing commonly used passwords. Then there is rule based attacks, where a website may require certain rules to be followed when composing a password (ie. "Your password must have at least 8 characters, one capital, and a number"). This can behave in an opposite manner to it's intended purpose where now attackers know a basic rule strucutre your password must follow.

My thinking was what if we could combine the two with Deep Learning. What if using a list of commonly used passwords we created 'rules'. Not real rules that have to be followed, but tendencies that people have in their passwords.

```bash
Michael8! Joe5$ Peter9( Bartholomew3@
```

All of these follow the same pattern, a name, number, and special character. However, some of these as a password are simple to crack (like Joe5$ with a password entropy of 32.85) whereas others are a bit tougher (like Bartholomew3@ with a password entropy of 85.41).

Now using a RNN trained on these passwords and generating passwords, how many of those can we find in a category of entropy higher.

## Features

- Password generation using RNN models.
- Comparison of generated passwords complexity and randomness.

## Installation

These installation steps are assuming a fresh WSL2 Instance. Some of these steps may already be taken care of so feel free to skip steps that have already been completed.

1. Install Linux on Windows with WSL

   This link has all the information you need.

   - https://learn.microsoft.com/en-us/windows/wsl/install

   Setup WSL virtual resources. In Powershell executing 'notepad.exe .\\.wslconfig' will open the configurations. Below is a sample.

   ```bash
   [wsl2]
   memory=110GB
   processors = 6
   ```

2. Set up CUDA for WSL2

   This link has the steps to do this.

   - https://docs.nvidia.com/cuda/wsl-user-guide/index.html
   - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

   I would recommend the deb (local) since the runfile (local) requires a specific NVIDIA Driver version, so if you already have a driver version of that number of above and do not wish to downgrade the deb is preferable.

   However, it does have the problem of potentially not updating the PATH variable. After the installation steps 'nvcc --version' still did not work even though I could see it existed and successfully call it with './nvcc --version' when I navigated to that directory.

   - To fix this add 'export PATH=/usr/local/cuda-12.2/bin:$PATH' to the end of your .bashrc file. This will update your PATH variable to include the path to the CUDA installation in addition to the original PATH variable.

3. Install Tensorflow

   This link has the specific steps to set this up:

   - https://www.tensorflow.org/install/pip#windows-wsl2

   When following these steps, a conda environment is created. This conda environment will be necessary to successfully run the code (or not but it seemed important when I did it).

4. Clone this repository to your local machine

   ```bash
   git clone https://github.com/MichaelGW123/PASSCODES.git
   cd PASSCODES
   ```

5. Install the required packages

   Make sure the conda environment is activated.

   ```bash
   pip install -r pip_requirements.txt
   ```

## Usage

Right now this simply is used to generate files and then we compare that, but this section will be expanded as the orignal implementation of generating the passwords is improved.

1. Using the training set, seperate it into different entropies of any desired granularity.

   - This step starts off with using the 'hashesorg2019' file. File_Analysis/basic_process.py will create a tsv, with each row containing [password (column 0), entropy (column 1)]. Entropy will be calculated by considering the unique pools of characters pulled from (ex. lowercase alphabetical, uppercase alphabetical, numerical, and special symbols) to assemble the character set then raised to the power of the length of the password, then the log2 computed. This will make it easier to modify later as we will not need to recalcuate the entropy.

     ```bash
     File_Analysis/basic_process.py
     ```

     TODO: Add diagram of different groups, and example.

   - After you have done File_Analysis/basic_process.py you can sort because there is an entropy calculated. Unix sort was used to sort the hashesorg2019.tsv

     ```bash
     sort -t$'\t' -k2,2n hashesorg2019.tsv > sorted_hashesorg2019.tsv
     ```

   - (OPTIONAL) File_Analysis/outliers.py will generate 2 TSVs based on the input TSV. It can be hard coded threshold for limit of mean + 6 \* std_dev (383 entropy) for the hashesorg2019.tsv file. It will separate those with entropy above that limit and those below into their respective TSV file. For the purpose of not removing passwords, this research will not remove any outliers except for the very largest password because it was so uniquely on it's own in its entropy value.

     ```bash
     head -n -1 sorted_hashesorg2019.tsv > temp && mv temp sorted_hashesorg2019.tsv
     ```

   - File_Analysis/plot.py (currently doesn't work because of memory constraints) will generate a KDE or Histogram of the entropy for the passwords to give a general idea of how the password entropy is distributed.

   - Unix split was used to split the dataset into files of increasing entropy (due to the fact it was already sorted). These will be the sets with which to train. In this case, 'wc -l' was used to get the number of lines in the file, then calculate a number of files which divides that number without any remainder.

     ```bash
     wc -l sorted_hashesorg2019.tsv
     split -l 25817639 sorted hashesorg2019.tsv entropy_data/entropy_bin
     ```

2. Determine how you want to test effectiveness. For example, you can test in the same entropy bin by using a test and training set, or aim for the entropy set above the one you train with (therefore allowing you to use 100% of the entropy bin below your target set)

3. Train your model on the desired set.

   TODO: Modify the code to work with both parameters passed from command line or whatever values were set in the code upon execution if no command line parameters we given.

   ```bash
   python PASSCODES.py entropy_bin_00 train
   ```

4. Generate a number of passwords with the RNN model.

   ```bash
   python PASSCODES.py entropy_bin_00 generate
   ```

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
