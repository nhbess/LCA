import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

import Visuals
np.set_printoptions(precision=2, suppress=True)

class LS:
    def __init__(self, n:int, m:int, n_symbols:int, production_rules = torch.Tensor) -> None:

        self.symbols = np.arange(n_symbols)
        self.P = self._convert_to_production_rules(production_rules)
        
        self.B = torch.zeros((n, m))  # Use torch.zeros instead of np.zeros
        self.B[n // 2, m // 2] = 1  # This line remains the same
        self.data = []
    
    def _convert_to_production_rules(self, array, abs=False) -> torch.Tensor:
        array = torch.clamp(array, -1, 1)
        if abs:
            array = torch.abs(array)
        sign = torch.sign(array)  # Use torch.sign
        P = torch.floor(torch.abs(array) * len(self.symbols)).to(torch.int)  # Use torch.floor and convert to int
        P[P == len(self.symbols)] = len(self.symbols) - 1  # No change needed
        P = (P * sign).to(torch.int)  # Use .to(torch.int) instead of .astype(int)
        P = P.reshape(-1, 2, 3, 3)
        P[:, 0] = torch.abs(P[:, 0])
        P[0, 0] = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0]])

        return P
    
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
        S = self.B
        N = np.zeros_like(S)
        for rule in self.P:
            reactant, product = rule
            matches = self._find_matches(S, reactant)
            if len(matches) > 0:
                N = N + self._replace_pattern(np.zeros_like(S), product, matches)
        
        self.B = self.B + N
        
        self.B = np.clip(self.B, 0, len(self.symbols)-1)
        
        self.data.append(self.B)

if __name__ == '__main__':
    pass
    seed = np.random.randint(0, 100000000) 
    #seed = 25576077 
    np.random.seed(seed)
    print(f'Seed: {seed}')
    Y = 10
    X = Y   #int(Y/ratio)
    RUNS = 50
    N_PRODUCTION_RULES = 3
    N_SYMBOLS = 2
    N_PARAMETERS =  N_PRODUCTION_RULES * 2 * 3 * 3

    P = torch.tensor(np.random.rand(N_PARAMETERS)*2 - 1, requires_grad=True)

    for run in range(1):
        b = LS(n=X, m=Y,n_symbols=N_SYMBOLS, production_rules=P)
        
        for i in range(Y*2):
            b.update()
            #print(b.B)
        data = b.data
        #data to numpy array
        data = np.array(data)
        Visuals.create_visualization_grid(data, filename=f'Test', duration=100, gif=True, video=False)