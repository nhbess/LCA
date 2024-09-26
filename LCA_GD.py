import torch
import numpy as np
import Util
import sys
from LSystem_Torch import LS

np.set_printoptions(precision=2, suppress=True)



def loss_function(params: torch.Tensor, 
                  target: torch.Tensor,
                  n_symbols: int,
                  n_updates: int) -> torch.Tensor:
    
    X, Y = target.shape
    b = LS(n=X, m=Y, n_symbols=n_symbols, production_rules=params)
    for _ in range(n_updates):
        b.update()
    
    if False:
        target = target.unsqueeze(0).repeat(n_updates, 1, 1)
        result = torch.tensor(b.data, device=target.device, dtype=torch.float32)
    else:
        result = torch.tensor(b.data[-1], device=target.device, dtype=torch.float32)
        result.requires_grad_(True) # Thi doesnt have sense, results is not a parameter to optimize FIX THIS
    
    loss = torch.sum((target - result) ** 2)  
    return loss


def train(target: torch.Tensor, 
          training_steps:int,
          n_symbols:int,
          n_updates:int,):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target, device=device)
    
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3 # reactants and products
    
    params = torch.randn(N_PARAMETERS, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([params], lr=0.01)

    for step in range(training_steps):
        optimizer.zero_grad()
        loss = loss_function(params, target, n_symbols, n_updates)
        loss.backward()
        optimizer.step()
        print(f'Step {step}, Loss: {loss.item()}') #Params: {params.detach().cpu().numpy()},

    return params.detach().cpu().numpy()
    
if __name__ == '__main__':
    
    base_folder = 'Face'
    target = Util.load_simple_image_as_numpy_array(f'__ASSETS/{base_folder}.png')
    
    N_SYMBOLS = 2
    N_PRODUCTION_RULES = 5
    N_UPDATES = 25
    
    rules = train(target=target, 
            training_steps=1000,
            n_symbols = N_SYMBOLS,
            n_updates = N_UPDATES,
            )
    
    print(rules)