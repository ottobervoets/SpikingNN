import pandas as pd
import os
import glob

path = '/Users/ottobervoets/Documents/23/MAI_ML_SNN/results'
extension = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(extension))

results = []
print(files)
for file in files:
    temp = pd.read_csv(file)
    acc = temp.iloc[4][1]
    acc = acc.replace('[', '')
    acc = acc.replace(']', '')
    acc = acc.split()
    acc = list(map(float, acc))
    
    acc.extend(temp.iloc[0:4,1].tolist())
    results.append(acc)
results = pd.DataFrame(results)
print(results.sort_values(0, ascending = False))
print(results)