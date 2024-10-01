import torch
import torch.nn.functional as F
import sys

#set print options
torch.set_printoptions(precision=2, sci_mode=False)



def print_gradients(tensor: torch.Tensor, where: str):
    if tensor.grad is not None:
        print(f'Gradient in {where}')
        tensor.grad
    else: print(f'No gradient in {where}')


class LS:
    def __init__(self, 
                 n: int, 
                 m: int,
                 production_rules: torch.Tensor,
                 n_production_rules: int, 
                 device='cpu') -> None:
        
        self.M = 1
        self.n_production_rules = n_production_rules
        
        self.production_rules = torch.clamp(production_rules, -1, 1)
        reactants, products = torch.split(self.production_rules, self.production_rules.shape[0] // 2)
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
        negatives_mask = (products <0).float()
        positives_mask = (products >0).float()
        products = positives_mask - negatives_mask
        return products

    def _handle_reactants(self, reactants: torch.Tensor) -> torch.Tensor:
        positive_mask = (reactants >0).float()
        reactants = positive_mask
        clean_mask = torch.ones_like(reactants)        
        clean_mask[0:9] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        seed_mask = torch.zeros_like(reactants)
        seed_mask[0:9] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0])
        reactants = reactants * clean_mask + seed_mask
        return reactants
    
    def update(self):
        new_board = torch.zeros_like(self.B, requires_grad=True)
        B_unsqueezed = self.B.unsqueeze(0).unsqueeze(0).float()  # Convert to float
        for i in range(self.n_production_rules):
            reactant = self.reactants[i].unsqueeze(0).unsqueeze(0)  # Ensure float
            reaction = F.conv2d(B_unsqueezed, reactant, padding=1)  # Use padding=1 for full matrix sliding

            sum_kernel = torch.sum(reactant)
            mask = (reaction == sum_kernel)  # Keep as boolean, no need to cast to float yet
            product = self.products[i].unsqueeze(0).unsqueeze(0)  # Ensure float tensors
            
            # Apply mask using multiplication, which should preserve gradients
            production = F.conv2d(mask.float(), product, padding=1)  # Convert to float here
            new_board = new_board + production.squeeze(0).squeeze(0)
            
        self.B = self.B + new_board
        self.B = torch.clamp(self.B, 0, 1)
        self.data.append(self.B.clone())
        return


   
if __name__ == '__main__':
    #make a simple test
    #set seed
    torch.manual_seed(15)
    n = 5
    m = 5
    n_symbols = 2
    n_production_rules = 4

    #production_rules = torch.randint(0, 2, (n_production_rules * 2 * 3 * 3,), dtype=torch.float32)*2-1
    production_rules = torch.randn(n_production_rules * 2 * 3 * 3, requires_grad=True)*2-1
    print(f'ls.production_rules: {production_rules}')
    ls = LS(n, m, production_rules, n_production_rules)
    print(f'ls.production_rules: {ls.production_rules}')
    print(f'ls.reactants\n{ls.reactants}')
    print(f'ls.products\n{ls.products}')


    ls.B[1, 1:3] = 1
    ls.B[2, 1:3] = 1
    ls.B[3, 1:3] = 1
    #ls.B = torch.ones_like(ls.B)
    print(f'Initial ls.B\n{ls.B}')
    
    for i in range(1):
        ls.update()
        #print(ls.B)
    pass