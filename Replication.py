import json
from LSystem import LS
import numpy as np

experiment_details_path = f'EXPS/Face2/Run_0/experiment_details.json'
rules_path = f'EXPS/Face2/Run_0/rules.txt'

with open(experiment_details_path, 'r') as f:
    experiment_details = json.load(f)

N_SYMBOLS= experiment_details['N_SYMBOLS']
N_PRODUCTION_RULES= experiment_details['N_PRODUCTION_RULES']
N_UPDATES= experiment_details['N_UPDATES']

rules = np.loadtxt(rules_path).reshape(-1, 2, 3, 3)
b = LS(n=50, m=50, n_symbols=N_SYMBOLS, production_rules=rules)

print(b.P)
for i in range(N_UPDATES):
    b.update()
data = b.data

print(data[-1])

import Visuals
Visuals.create_visualization_grid(data=data, filename=f'animation', duration=100, gif=True, video=False)
