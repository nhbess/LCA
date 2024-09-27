import torch
import numpy as np
import Util
import sys
from LSystem_Torch import LS

np.set_printoptions(precision=2, suppress=True)

def loss_function(ls :LS,
                  params: torch.Tensor, 
                  target: torch.Tensor,
                  n_updates: int) -> torch.Tensor:
    
    ls.reset(params)
    for _ in range(n_updates):
        ls.update()
    
    ls.B.requires_grad = True #WTF?

    loss = torch.sum((target - ls.B) ** 2)
    
    return loss


def train(target: torch.Tensor, 
          training_steps: int,
          n_symbols: int,
          n_updates: int):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target, device=device, dtype=torch.float32)
    X, Y = target.shape
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3  # reactants and products    
    params = torch.randn(N_PARAMETERS, requires_grad=True, device=device)

    print(f'params.grad: {params.grad} params.requires_grad: {params.requires_grad}')

    optimizer = torch.optim.Adam([params], lr=0.01)

    ls = LS(X, Y, n_symbols, n_production_rules=N_PRODUCTION_RULES, production_rules=params, device=device)    

    for step in range(training_steps):

        optimizer.zero_grad()
        loss = loss_function(ls, params, target, n_updates)
        loss.backward()
        optimizer.step()

        print(f'Step {step}, Loss: {loss.item()}')

    return params.detach().cpu().numpy()
    
if __name__ == '__main__':
    
    base_folder = 'Face'
    target = Util.load_simple_image_as_numpy_array(f'__ASSETS/{base_folder}.png')
    
    N_SYMBOLS = 2
    N_PRODUCTION_RULES = 5
    N_UPDATES = 25
    
    rules = train(target=target, 
            training_steps=10,
            n_symbols = N_SYMBOLS,
            n_updates = N_UPDATES,
            )
    
    print(rules)