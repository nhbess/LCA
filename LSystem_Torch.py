import torch
import sys

class LS:
    def __init__(self, 
                 n: int, 
                 m: int,
                 production_rules: torch.Tensor,
                 n_production_rules: int, 
                 device='cpu') -> None:
        
        self.M = 1
        self.n_production_rules = n_production_rules
        self.production_rules = production_rules
        
        self.production_rules = torch.clamp(self.production_rules, -1, 1)
        reactants, products = torch.split(production_rules, production_rules.shape[0] // 2)
        self.reactants = self._handle_reactants(reactants).view(n_production_rules, 3, 3)
        self.products = self._handle_products(products).view(n_production_rules, 3, 3)


        board = torch.zeros((n, m), dtype=torch.float32, requires_grad=True)

        seed_mask = torch.zeros_like(board)
        seed_mask[board.shape[0]//2, board.shape[1]//2] = 1
        self.B = board + seed_mask

        self.data = []

        self.device = device
        self.B = self.B.to(self.device)
        self.reactants = self.reactants.to(self.device)
        self.products = self.products.to(self.device)

    def _handle_products(self, products: torch.Tensor) -> torch.Tensor:
        negatives = torch.relu(-products)
        positives = torch.relu(products)
        zeros     = torch.exp(-products*products)
        signs     = torch.tanh(products)
        mapped_products = torch.sum(torch.stack([negatives, -zeros, positives]), dim=0)*signs
        return mapped_products

    def _handle_reactants(self, reactants: torch.Tensor) -> torch.Tensor:
        negatives = torch.relu(-reactants)
        positives = torch.relu(reactants)
        zeros     = torch.exp(-reactants*reactants)
        signs     = torch.tanh(reactants)
        mapped_reactants = torch.sum(torch.stack([-zeros, positives]), dim=0)
        mapped_reactants[0:9] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
        return mapped_reactants


    def update(self):
        new_board = torch.zeros_like(self.B, requires_grad=True)
        for i in range(self.B.shape[0] - self.reactants[0].shape[0] + 1):
            for j in range(self.B.shape[1] - self.reactants[0].shape[1] + 1):
                subgrid = self.B[i:i + self.reactants[0].shape[0], j:j + self.reactants[0].shape[1]]
                for k in range(self.reactants.size(0)):
                    error_matrix = torch.abs(subgrid - self.reactants[k])
                    total_error = torch.sum(error_matrix)
                    corrects = torch.exp(-total_error*total_error*self.M/2)

                    update_board = torch.zeros_like(new_board)
                    update_board[i:i+self.reactants[k].shape[0], j:j+self.reactants[k].shape[1]] = self.products[k] * corrects
                    
                    new_board = new_board + update_board
            
        self.B = self.B + new_board
        self.B = torch.clamp(self.B, 0, 1)
        self.data.append(self.B.clone())
        return

if __name__ == '__main__':
    #make a simple test
    n = 5
    m = 5
    n_symbols = 2
    n_production_rules = 4

    production_rules = torch.rand(n_production_rules * 2 * 3 * 3) * 2 - 1
    ls = LS(n, m, production_rules, n_production_rules)
    ls.update()
    print(ls.B)
    pass