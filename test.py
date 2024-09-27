import torch
torch.set_printoptions(precision=1, sci_mode=False)

n_production_rules = 2

production_rules = torch.rand(n_production_rules * 2 * 3 * 3, requires_grad=True) * 2 - 1
print(f'production_rules:\n{production_rules}')

reactants, products = torch.split(production_rules, production_rules.shape[0] // 2)
#print(f'reactants:\n{reactants}')


# PRODUCTS
print('PRODUCTS ----------------')
print(f'products:\n{products}')

#map it to -1,0,1 
M = 1000
negatives = 1/(1+torch.exp(products*M))
positives = 1/(1+torch.exp(-products*M))
zeros     = torch.exp(-products*products*M/2)
signs     = torch.tanh(products*M)

print(f'negatives:\n{negatives}')
print(f'positives:\n{positives}')
print(f'zeros:\n{zeros}')
print(f'signs:\n{signs}')

mapped_products = torch.sum(torch.stack([negatives, -zeros, positives]), dim=0)*signs
print(f'mapped_products:\n{mapped_products}')

# REACTANTS
print('REACTANTS ----------------')
print(f'reactants:\n{reactants}')
positives = 1/(1+torch.exp(-reactants*M))
negatives = 1/(1+torch.exp(reactants*M))
zeros     = torch.exp(-reactants*reactants*M/2)
signs     = torch.tanh(reactants*M)

print(f'negatives:\n{negatives}')
print(f'positives:\n{positives}')
print(f'zeros:\n{zeros}')
print(f'signs:\n{signs}')

mapped_reactants = torch.sum(torch.stack([-zeros, positives]), dim=0)
print(f'mapped_reactants:\n{mapped_reactants}')

# BOARD
print('BOARD ----------------')
board = torch.rand((4, 4), dtype=torch.float32, requires_grad=True)*2-1
board = 1/(1+torch.exp(-board*M))
print(f'board:\n{board}')

new_board = torch.zeros_like(board, requires_grad=True)
# i need to read all the subgrids of the board and compare them with the reactants

mapped_reactants = mapped_reactants.view(1, 2, 3, 3)
mapped_products = mapped_products.view(1, 2, 3, 3)
print(f'mapped_reactants:\n{mapped_reactants}')
print(f'mapped_products:\n{mapped_products}')
for i in range(board.shape[0] - 2):
    for j in range(board.shape[1] - 2):
        subgrid = board[i:i+3, j:j+3]
        #print(f'subgrid:\n{subgrid}')
        for k in range(len(mapped_reactants[0])):
            continue
            reactant = mapped_reactants[0][k]
            product = mapped_products[0][k]

            #print(f'reactant:\n{reactant}')
            error_matrix = torch.abs(subgrid - reactant)
            #print(f'error_matrix:\n{error_matrix}') 
            total_error = torch.sum(error_matrix)
            total_error = torch.exp(-total_error*total_error*M/2)
            new_board[i:i+3, j:j+3] += total_error
            print(f'total_error:\n{total_error}')