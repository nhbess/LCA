import os
import json
import sys
import numpy as np
from Evolution import es
import Visuals
from LSystem import LS
import Util
import pickle

np.set_printoptions(precision=2, suppress=True)



def _reward_function_individual(individual:np.array, target:np.array, n_symbols:int, n_updates:int) -> float: 
    #print(target)
    #print(individual)

    # Create LSystem with individual production rules
    X,Y = target.shape
    rules = np.copy(individual)
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]

    b = LS(n=X, m=Y, n_symbols=n_symbols, production_rules=rules)
    for i in range(n_updates):
            b.update()
    
    result = b.data
    loss = np.sum(np.square(target - result[-1]))
    reward = 1 / (1 + loss)
    return reward

def evolve(target:np.array, num_params:int, n_symbols:int, n_updates:int, n_generations=100, popsize=20, folder:str = 'test'):
    
    solver = es.CMAES(num_params=num_params, popsize=popsize, weight_decay=0.01, sigma_init=0.5)
    results = {'BEST': [],'REWARDS': []}
    
    for g in range(n_generations):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)

        for i in range(solver.popsize):
            fitness_list[i] = _reward_function_individual(np.array(solutions[i]), target, n_symbols, n_updates)
            
        solver.tell(fitness_list)
        result = solver.result()
        
        best_params, best_reward, curr_reward, sigma = result[0], result[1], result[2], result[3]
        print(f'G:{g}, BEST PARAMS, BEST REWARD: {best_reward}, CURRENT REWARD: {curr_reward}')
        
        results['BEST'].append(best_params.tolist())
        results['REWARDS'].append(fitness_list.tolist())

        this_dir = os.path.dirname(os.path.abspath(__file__))        
        file_path = os.path.join(this_dir, f'{folder}/results.json')
    
    with open(file_path, 'w') as f:
        json.dump(results, f)

    return best_params

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000)
    np.random.seed(seed)
    
    target = Util.load_image_as_numpy_array('Mario.png', black_and_white=True, binary=True, sensibility=0.1)
    
    X,Y = target.shape
    
    N_PRODUCTION_RULES = 10
    N_SYMBOLS = 3
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3 # reactants and products
    POP_SIZE = 10
    N_GENERATIONS = 50
    N_UPDATES = 15

    folder_path = f'PR{N_PRODUCTION_RULES}POP{POP_SIZE}GEN{N_GENERATIONS}'
    os.makedirs(folder_path, exist_ok=True)    

    best_individual = evolve(target=target, 
                             n_symbols=N_SYMBOLS,
                             n_updates = N_UPDATES,
                             num_params=N_PARAMETERS,
                             popsize=POP_SIZE,
                             n_generations=N_GENERATIONS,
                             folder=folder_path)
    

    rules = np.copy(best_individual)
    np.savetxt(f'{folder_path}/rules.txt', rules.flatten())
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]
    #save model b


    b = LS(n=X, m=Y, n_symbols=N_SYMBOLS, production_rules=rules)
    #save model b in a pickle file
    with open(f'{folder_path}/model.pkl', 'wb') as f:
        pickle.dump(b, f)

    print(f'P: {b.P}')
    
    for i in range(X):
        b.update()


    data = b.data
    print(data[-1])
    Visuals.create_visualization_grid(data, filename=f'{folder_path}/animation', duration=100, gif=True, video=False)
    Visuals.visualize_target_result(target, data, filename=f'{folder_path}/Result.png')
    Visuals.visualize_evolution_results(result_path=f'{folder_path}/results.json', filename=f'{folder_path}/Best_rewards.png')

