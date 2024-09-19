import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image
from scipy.ndimage import convolve
#from tqdm import tqdm
import Visuals
np.set_printoptions(precision=2, suppress=True)

class LS:
    def __init__(self, n:int, m:int, n_symbols:int, n_production_rules:int = 2, production_rules = None) -> None:
        self.symbols = np.arange(n_symbols)
        
        if production_rules is not None:    self.P = self._discretize_production_rules(production_rules)
        else:                               self.P = self._make_production_rules(n_production_rules)
        
        self.P[0] = [np.array([[0, 0, 0], 
                               [0, 1, 0], 
                               [0, 0, 0]]), 
                                self.P[0][1]]   #First rule to match the seed

        self.B = np.zeros((n,m), dtype=int)
        self.B[n//2, m//2] = 1
        self.data = []

    def _discretize_production_rules(self, production_rules) -> list:
        def map_to_symbols(array, abs=False):
            if abs: array = np.abs(array)  # Take the absolute value
            sign = np.sign(array)
            mapped = np.floor(np.abs(array) * len(self.symbols)).astype(int)
            mapped[mapped == len(self.symbols)] = len(self.symbols) - 1  # To handle the edge case when array value is exactly 1
            mapped = (mapped * sign).astype(int)
            return mapped
        
        P = [[map_to_symbols(rule[0],abs=True), map_to_symbols(rule[1])] for rule in production_rules]
        return P

    def _make_production_rules(self, n_production_rules) -> dict:
        P = []
        P.append([  np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [0, 0, 0]]), 
                    np.random.rand(3,3)*2-1]) #First rule

        for _ in range(n_production_rules):
            reactants = np.random.rand(3,3)*2-1
            products = np.random.rand(3,3)*2-1
            P.append([reactants, products])

        return self._discretize_production_rules(P)
    
    def _find_matches(self, S:np.array, reactant:np.array) -> list:
        M_rows, M_cols = S.shape
        m_rows, m_cols = reactant.shape

        matches = []
        for i in range(M_rows - m_rows + 1):
            for j in range(M_cols - m_cols + 1):
                subgrid = S[i:i+m_rows, j:j+m_cols]        
                if np.array_equal(subgrid, reactant):
                    matches.append((i, j))
        return matches

    def _replace_pattern(self, M, m, matches):
        m_rows, m_cols = m.shape
        for match in matches:
            i, j = match
            M[i:i+m_rows, j:j+m_cols] = m        
        return M

    def update(self) -> None:
        S = self.B.copy()
        N = np.zeros_like(S)
        for rule in self.P:
            reactant, product = rule
            matches = self._find_matches(S, reactant)
            if len(matches) > 0:
                N = N + self._replace_pattern(np.zeros_like(S), product, matches)
        
        self.B = self.B + N
        self.B = np.clip(self.B, 0, len(self.symbols)-1) 
        self.data.append(self.B.copy())

if __name__ == '__main__':
    pass
    seed = np.random.randint(0, 100000000) 
    #seed = 25576077 
    np.random.seed(seed)
    print(f'Seed: {seed}')
    Y = 10
    X = Y   #int(Y/ratio)
    RUNS = X
    N_PRODUCTION_RULES = 20
    N_SYMBOLS = 5

    for run in range(1):
        b = LS(n=X, m=Y,n_symbols=N_SYMBOLS, n_production_rules=N_PRODUCTION_RULES)
        for i in range(Y*2):
            b.update()

        data = b.data
        Visuals.create_visualization_grid(data, filename=f'Test', duration=100, gif=True, video=False)