import torch

class LS:
    def __init__(self, n: int, m: int, n_symbols: int, n_production_rules: int = 2, production_rules=None, device='cpu') -> None:
        self.symbols = torch.arange(n_symbols, device=device)
        
        if production_rules is not None:
            self.P = self._map_to_symbols(production_rules).view(-1, 2, 3, 3)
        else:
            self.P = self._make_production_rules(n_production_rules)
        
        # Ensure the first rule is set manually
        self.P[0] = torch.stack([torch.tensor([[0, 0, 0], 
                                               [0, 1, 0], 
                                               [0, 0, 0]], dtype=torch.float32, device=device), 
                                 self.P[0][1]]) #First rule to match the seed
        
        # Ensure reactants are positive
        self.P[:, 0] = torch.abs(self.P[:, 0])

        self.B = torch.zeros((n, m), dtype=torch.float32, device=device)
        self.B[n // 2, m // 2] = 1
        self.data = []

    def _map_to_symbols(self, array, abs=False):
        array = torch.clamp(array, -1, 1)
        if abs:
            array = torch.abs(array)
        sign = torch.sign(array)
        mapped = torch.floor(torch.abs(array) * len(self.symbols)).long()
        mapped[mapped == len(self.symbols)] = len(self.symbols) - 1
        mapped = (mapped * sign).long()
        return mapped

    def _make_production_rules(self, n_production_rules) -> torch.Tensor:
        n_parameters = n_production_rules * 2 * 3 * 3  # reactants and products
        production_rules = torch.rand(n_parameters, device=self.symbols.device) * 2 - 1  # Random values between -1 and 1
        production_rules[0] = 1.0  # Set a specific value for the first rule
        P = self._map_to_symbols(production_rules).view(-1, 2, 3, 3)
        return P

    def _find_matches(self, S: torch.Tensor, reactant: torch.Tensor) -> list:
        M_rows, M_cols = S.shape
        m_rows, m_cols = reactant.shape
        matches = []
        for i in range(M_rows - m_rows + 1):
            for j in range(M_cols - m_cols + 1):
                subgrid = S[i:i+m_rows, j:j+m_cols]
                # Use an approximate equality check to keep gradients
                if torch.allclose(subgrid, reactant):
                    matches.append((i, j))
        return matches

    def _replace_pattern(self, M, m, matches):
        m_rows, m_cols = m.shape
        for match in matches:
            i, j = match
            M[i:i+m_rows, j:j+m_cols] = m
        return M

    def update(self) -> None:
        S = self.B.clone()  # Use torch.clone() instead of np.copy()
        N = torch.zeros_like(S)
        for rule in self.P:
            reactant, product = rule
            matches = self._find_matches(S, reactant)
            if len(matches) > 0:
                N = N + self._replace_pattern(torch.zeros_like(S), product, matches)
        
        self.B = self.B + N
        
        # Adjust clipping for symbols in torch
        ALLOW_NEGATIVE = False
        if ALLOW_NEGATIVE:
            self.B = torch.clamp(self.B, -(len(self.symbols)-1), len(self.symbols)-1)
        else:
            self.B = torch.clamp(self.B, 0, len(self.symbols)-1)
        
        self.data.append(self.B.clone())  # Save a copy of the board state
