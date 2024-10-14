import json
from LSystem import LS
import numpy as np
from tqdm import tqdm
ID = 6
FOLDER = 'GoodFace'
experiment_details_path = f'{FOLDER}//Run_{ID}//experiment_details.json'
rules_path = f'{FOLDER}//Run_{ID}//rules.txt'

with open(experiment_details_path, 'r') as f:
    experiment_details = json.load(f)

N_SYMBOLS= experiment_details['N_SYMBOLS']
N_PRODUCTION_RULES= experiment_details['N_PRODUCTION_RULES']
N_UPDATES= experiment_details['N_UPDATES']

rules = np.loadtxt(rules_path).reshape(-1, 2, 3, 3)

GRID_SIZE = 128*2
b = LS(n=GRID_SIZE, m=GRID_SIZE, n_symbols=N_SYMBOLS, production_rules=rules)

print(b.P)
print('Starting update')
for i in tqdm(range(int(N_UPDATES*5))):
    b.update()

data = b.data
n_frames = len(data)
n_frames = 6
N = int(np.ceil(np.sqrt(n_frames)))  # ceil to ensure we have enough subplots
M = int(np.ceil(n_frames / N))

#make a plot for each frame
if False:
    import matplotlib.pyplot as plt
    print('Starting plot')
    fig, ax = plt.subplots(M, N, figsize=(N, M))
    for i in range(n_frames):
        ax[i//N, i%N].imshow(data[i], cmap='gray')
    #set all the ticks to be invisible
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
    plt.show()


import Visuals
filename=f'animation_{FOLDER}_{GRID_SIZE}'

print('Starting visualization')

Visuals.visual_perfect_pixel(data, filename)
#Visuals.create_visualization_grid(data=data, filename=f'animation_{FOLDER}_{GRID_SIZE}', duration=100, gif=True, video=False)

print('Done')