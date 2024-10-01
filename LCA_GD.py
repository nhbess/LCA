import torch
import numpy as np
import Util
import sys
from LSystem_Torch import LS
from torch.optim.lr_scheduler import MultiStepLR

np.set_printoptions(precision=2, suppress=True)

def loss_function(params: torch.Tensor, 
                  target: torch.Tensor,
                  n_updates: int) -> torch.Tensor:
    
    n,m = target.shape
    ls = LS(n=n, m=m, production_rules=params, n_production_rules=N_PRODUCTION_RULES, device = params.device)
    #print(f'params:\n{params}')
    for _ in range(n_updates):
        ls.update()
    
    #print(f'target:\n{target}')
    #print(f'ls.B:\n{ls.B}')
    loss = torch.sum((target - ls.B) ** 2)
    
    return loss


def train(target: torch.Tensor, 
          training_steps: int,
          n_updates: int):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    target = torch.tensor(target, device=device, dtype=torch.float32)
    X, Y = target.shape
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3  # reactants and products    

    production_rules = torch.randn(N_PARAMETERS, requires_grad=True, device=device)

    gamma = 0.4
    lr_max = 0.1/gamma
    n_milestones = 4
    milestones = np.linspace(0, training_steps, n_milestones, endpoint=False, dtype=int)
    LR = 0.01
    optimizer = torch.optim.Adam([production_rules], lr=lr_max, weight_decay=0.0)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
        
    LOSSES = []
    for step in range(training_steps):
        #print current learning rate

        optimizer.zero_grad()
        loss = loss_function(production_rules, target, n_updates)
        loss.backward()
        #if production_rules.grad is not None:
        #    production_rules.grad.data[torch.isnan(production_rules.grad.data)] = 1.0
            #production_rules.grad.data = torch.clamp(production_rules.grad.data, min=-1.0, max=1.0)
            #print(f'grad: {production_rules.grad.data}')


        optimizer.step()
        scheduler.step()

        LOSSES.append(loss.item())
        print(f'Step {step}, Loss: {loss.item()}, Learning rate: {scheduler.get_last_lr()}')

    return production_rules, LOSSES
    
if __name__ == '__main__':
    
    base_folder = 'Face2'
    target = Util.load_simple_image_as_numpy_array(f'__ASSETS/{base_folder}.png')
    
    N_PRODUCTION_RULES = 10
    N_UPDATES = 16
    
    rules, losses = train(target=target, 
            training_steps=300,
            n_updates = N_UPDATES,
            )
    n,m = target.shape

    ls = LS(n=n, m=m, production_rules=rules, n_production_rules=N_PRODUCTION_RULES)
    for _ in range(N_UPDATES):
        ls.update()


    data = [d.detach().cpu().numpy() for d in ls.data]
    
    import Visuals
    Visuals.create_visualization_grid(data, filename=f'GD_animation', duration=100, gif=True, video=False)
    Visuals.visualize_target_result(target, data, filename=f'GD_Result.png')
    #plot losses
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('GD_losses.png')
    plt.show()