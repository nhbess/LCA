import torch
n_production_rules = 1

reactants = torch.rand(n_production_rules * 3 * 3) * 2 - 1
products = torch.rand(n_production_rules * 3 * 3) * 2 - 1