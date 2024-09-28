import torch
import sys

torch.set_printoptions(precision=1, sci_mode=False)

M = 1000

def handle_products(products: torch.Tensor) -> torch.Tensor:
    negatives = 1/(1+torch.exp(products*M))
    positives = 1/(1+torch.exp(-products*M))
    zeros     = torch.exp(-products*products*M/2)
    signs     = torch.tanh(products*M)
    mapped_products = torch.sum(torch.stack([negatives, -zeros, positives]), dim=0)*signs
    return mapped_products

def handle_reactants(reactants: torch.Tensor) -> torch.Tensor:
    positives = 1/(1+torch.exp(-reactants*M))
    negatives = 1/(1+torch.exp(reactants*M))
    zeros     = torch.exp(-reactants*reactants*M/2)
    signs     = torch.tanh(reactants*M)
    mapped_reactants = torch.sum(torch.stack([-zeros, positives]), dim=0)
    mapped_reactants[0:9] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
    return mapped_reactants

def update(board:torch.tensor, reactants:torch.tensor, products:torch.tensor):
    new_board = torch.zeros_like(board, requires_grad=True)
    for i in range(board.shape[0] - reactants[0].shape[0] + 1):
        for j in range(board.shape[1] - reactants[0].shape[1] + 1):
            subgrid = board[i:i + reactants[0].shape[0], j:j + reactants[0].shape[1]]
            for k in range(reactants.size(0)):
                reactant = reactants[k]
                product = products[k]                
                error_matrix = torch.abs(subgrid - reactant)
                total_error = torch.sum(error_matrix)
                total_error = torch.exp(-total_error*total_error*M/2)

                update_board = torch.zeros_like(new_board)
                update_board[i:i+reactant.shape[0], j:j+reactant.shape[1]] = products[k] * total_error
                
                # Add the update_board to new_board (out-of-place accumulation)
                new_board = new_board + update_board
        
        board = board + new_board
        board = torch.clamp(board, 0, 1)
    return board

n_production_rules = 2

production_rules = torch.rand(n_production_rules * 2 * 3 * 3, requires_grad=True) * 2 - 1
reactants, products = torch.split(production_rules, production_rules.shape[0] // 2)

mapped_reactants = handle_reactants(reactants).view(n_production_rules, 3, 3)
mapped_products = handle_products(products).view(n_production_rules, 3, 3)

print(f'mapped_reactants:\n{mapped_reactants}')
print(f'mapped_products:\n{mapped_products}')


print('BOARD ----------------')
board = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
seed_mask = torch.zeros_like(board)
seed_mask[board.shape[0]//2, board.shape[1]//2] = 1
board = board + seed_mask

print(f'board:\n{board}')

updated_board = update(board, mapped_reactants, mapped_products)
print(f'updated_board:\n{updated_board}')