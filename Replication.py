import json
from LSystem import LS
import numpy as np

ID = 0
experiment_details_path = f'LilFace//Run_{ID}//experiment_details.json'
rules_path = f'LilFace//Run_{ID}//rules.txt'

with open(experiment_details_path, 'r') as f:
    experiment_details = json.load(f)

N_SYMBOLS= experiment_details['N_SYMBOLS']
N_PRODUCTION_RULES= experiment_details['N_PRODUCTION_RULES']
N_UPDATES= experiment_details['N_UPDATES']

rules = np.loadtxt(rules_path).reshape(-1, 2, 3, 3)
b = LS(n=6, m=6, n_symbols=N_SYMBOLS, production_rules=rules)

print(b.P)
for i in range(N_UPDATES):
    b.update()

data = b.data
n_frames = len(data)
N = int(np.ceil(np.sqrt(n_frames)))  # ceil to ensure we have enough subplots
M = int(np.ceil(n_frames / N))

#make a plot for each frame
import matplotlib.pyplot as plt

fig, ax = plt.subplots(M, N, figsize=(N, M))
for i in range(n_frames):
    ax[i//N, i%N].imshow(data[i], cmap='gray')
#set all the ticks to be invisible
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
plt.show()



import Visuals
Visuals.create_visualization_grid(data=data, filename=f'animation', duration=100, gif=True, video=False)
