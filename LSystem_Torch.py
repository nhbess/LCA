import torch
import sys

class LS:
    def __init__(self, 
                 n: int, 
                 m: int, 
                 n_symbols: int, 
                 n_production_rules: int = 2, 
                 production_rules=None, 
                 device='cpu') -> None:
        
        self.symbols = torch.arange(n_symbols, device=device)
        
        self.P = self._map_to_symbols(production_rules).view(-1, 2, 3, 3)
        self._correct_P()

        self.B = torch.zeros((n, m), dtype=torch.float32, device=device) + 1e-8
        self.B[n // 2, m // 2] = 1
        self.data = []
    
    def _correct_P(self):
        self.P[0, 0] = torch.tensor([[0, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 0]], dtype=self.P.dtype, device=self.P.device)
        
        self.P[:, 0] = torch.abs(self.P[:, 0])
        return
    
    def _map_to_symbols(self, array):
        array = torch.clamp(array, -1, 1)
        sign = torch.sign(array)
        mapped = torch.floor(torch.abs(array) * len(self.symbols)).long()
        mapped[mapped == len(self.symbols)] = len(self.symbols) - 1
        mapped = (mapped * sign).long()
        return mapped

    def _make_production_rules(self, n_production_rules) -> torch.Tensor:
        n_parameters = n_production_rules * 2 * 3 * 3  # reactants and products
        production_rules = torch.rand(n_parameters, device=self.symbols.device) * 2 - 1  # Random values between -1 and 1
        production_rules[0] = 1.0  # Set a specific value for the first rule
        P = self._map_to_symbols(production_rules)
        return P

    def _find_matches(self, S: torch.Tensor, reactant: torch.Tensor) -> list:
        M_rows, M_cols = S.shape
        m_rows, m_cols = reactant.shape
        matches = []

        for i in range(M_rows - m_rows + 1):
            for j in range(M_cols - m_cols + 1):
                subgrid = S[i:i+m_rows, j:j+m_cols]
                # Use an approximate equality check to keep gradients
                if torch.allclose(subgrid.to(reactant.dtype), reactant):
                    matches.append((i, j))
        return matches

    def _replace_pattern(self, M, m, matches):
        m_rows, m_cols = m.shape
        for match in matches:
            i, j = match
            M[i:i+m_rows, j:j+m_cols] = m
        return M
    
    
    def reset(self, production_rules):
        self.B = torch.zeros_like(self.B, device=self.B.device)
        self.B[self.B.shape[0] // 2, self.B.shape[1] // 2] = 1
        self.data = []
        self.P = self._map_to_symbols(production_rules).view(-1, 2, 3, 3)
        self._correct_P()

    
    def update(self) -> None:
        N = torch.zeros_like(self.B)
        for rule in self.P:
            reactant, product = rule
            matches = self._find_matches(self.B, reactant)
            if len(matches) > 0:
                N = N + self._replace_pattern(torch.zeros_like(self.B), product, matches)
        
        self.B = self.B + N
        self.B = torch.clamp(self.B, 0, len(self.symbols)-1)
        
        self.data.append(self.B.clone())  # Save a copy of the board state

if __name__ == '__main__':
    #make a simple test
    n = 5
    m = 5
    n_symbols = 2
    n_production_rules = 2

    production_rules = torch.rand(n_production_rules * 2 * 3 * 3) * 2 - 1
    ls = LS(n, m, n_symbols, n_production_rules, production_rules)

    ls.update()
    print(ls.B)
    pass