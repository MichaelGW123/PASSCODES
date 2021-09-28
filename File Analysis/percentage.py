# Calculating percent of file found in other file

# Importing Libraries

from matplotlib import pyplot as plt
import math
import numpy as np
from pathlib import Path

testerFlag = True

tp = Path(__file__).with_name('Target.txt')
t = open(tp, 'r')
words=set(line.strip() for line in t)
targetNumber = len(words)
t.close()
if testerFlag:
    print(f'Target File Unique Occurences Length: {targetNumber}')

arrayPercentages = []
found = 0
percent = found/targetNumber

gp = Path(__file__).with_name('Generated.txt')
g = open(gp, 'r')
generatedLength = len(g.readlines())
g.close()
g = open(gp,'r')
for line in g.readlines():
    temp = line.strip()
    if temp in words:
        found += 1
        words.remove(temp)
        percent = found/targetNumber
        arrayPercentages.append(percent)
    else:
        arrayPercentages.append(percent)
g.close()
if testerFlag:
    print(f'Generated File Length: {generatedLength}')

if testerFlag:
    print(f'Array of Percentages: {arrayPercentages}')
genList = list(range(1, generatedLength+1))
if testerFlag:
    print(f'Array of Guesses: {genList}')

x = genList
y = arrayPercentages
plt.xlabel("Number of Generated Guesses")
plt.ylabel("Percent Matching Target File")
plt.title('Generated Matching Effectiveness')
plt.plot(x,y)
plt.show()