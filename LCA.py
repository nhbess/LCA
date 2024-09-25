import os
import json
import sys
import numpy as np
from Evolution import es
import Visuals
from LSystem import LS
import Util
import pickle
import argparse
import time

np.set_printoptions(precision=2, suppress=True)
def _reward_function_individual(individual:np.array, target:np.array, n_symbols:int, n_updates:int) -> float: 
    
    X,Y = target.shape
    rules = np.copy(individual)
    b = LS(n=X, m=Y, n_symbols=n_symbols, production_rules=rules)

    for _ in range(n_updates):
        b.update()
    
    result = b.data
    #only ones in results
    #result = np.array([np.where(r == 1, 1, 0) for r in result])

    loss = np.mean(np.square(target - result[-1])) # L2 loss
    #loss = np.sum(np.square(result - target), axis=1).sum()  #Accumulated L2 loss
    reward = -loss
    return reward



def evolve(target:np.array, num_params:int, n_symbols:int, n_updates:int, n_generations=100, popsize=20, folder:str = 'test'):
    
    solver = es.CMAES(num_params=num_params, popsize=popsize, weight_decay=0.01, sigma_init=0.5)
    results = {'REWARDS': []}
    
    for g in range(n_generations):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)

        for i in range(solver.popsize):
            fitness_list[i] = _reward_function_individual(np.array(solutions[i]), target, n_symbols, n_updates)
            
        solver.tell(fitness_list)
        result = solver.result()
        
        best_params, best_reward, curr_reward, sigma = result[0], result[1], result[2], result[3]
        print(f'G:{g}, BEST PARAMS, BEST REWARD: {best_reward}, CURRENT REWARD: {curr_reward}')
        
        #results['BEST'].append(best_params.tolist())
        results['REWARDS'].append(fitness_list.tolist())

        this_dir = os.path.dirname(os.path.abspath(__file__))        
        file_path = os.path.join(this_dir, f'{folder}/evolution_rewards.json')
    
    with open(file_path, 'w') as f:
        json.dump(results, f)

    return best_params

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000)
    np.random.seed(seed)

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Define the parameters with default values
    parser.add_argument('--n_symbols',          type=int, default=2,        help='Number of symbols')
    parser.add_argument('--n_production_rules', type=int, default=10,       help='Number of production rules')
    parser.add_argument('--pop_size',           type=int, default= 10,      help='Population size')
    parser.add_argument('--n_generations',      type=int, default= 20,      help='Number of generations')
    parser.add_argument('--n_updates',          type=int, default=30,       help='Number of updates')

    # Parse the arguments
    args = parser.parse_args()

    # Assign the parsed values to variables
    N_SYMBOLS = args.n_symbols
    N_PRODUCTION_RULES = args.n_production_rules
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.n_generations
    N_UPDATES = args.n_updates

    #target = Util.load_image_as_numpy_array('Mario.png', black_and_white=True, binary=False, sensibility=0.1)
    #target = Util.discretize_target(target, N_SYMBOLS)
    
    target = np.zeros((7,7))
    #make a cross
    target[:, target.shape[0]//2] = 1
    target[target.shape[1]//2, :] = 1
    print(target)

    X,Y = target.shape
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3 # reactants and products




    # SET FOLDERS
    base_folder = 'EXP'
    os.makedirs(base_folder, exist_ok=True)
    existing_runs = [int(d[4:]) for d in os.listdir(base_folder) if d.startswith('Run_') and d[4:].isdigit()]
    next_id = max(existing_runs, default=0) + 1
    folder_path = os.path.join(base_folder, f'Run_{next_id}')
    os.makedirs(folder_path)


    starting_time = time.time()
    best_individual = evolve(target=target, 
                             n_symbols=N_SYMBOLS,
                             n_updates = N_UPDATES,
                             num_params=N_PARAMETERS,
                             popsize=POP_SIZE,
                             n_generations=N_GENERATIONS,
                             folder=folder_path)
    
    end_time = time.time()

    experiment_details = {
        'N_SYMBOLS': N_SYMBOLS,
        'N_PRODUCTION_RULES': N_PRODUCTION_RULES,
        'POP_SIZE': POP_SIZE,
        'N_GENERATIONS': N_GENERATIONS,
        'N_UPDATES': N_UPDATES,
        'SEED': seed,
        'TIME': (end_time - starting_time) / 3600
    }
    
    with open(f'{folder_path}/experiment_details.json', 'w') as f:
        json.dump(experiment_details, f)
    rules = np.copy(best_individual)
    np.savetxt(f'{folder_path}/rules.txt', rules.flatten())
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]


    b = LS(n=X, m=Y, n_symbols=N_SYMBOLS, production_rules=rules)
    with open(f'{folder_path}/model.pkl', 'wb') as f:
        pickle.dump(b, f)

    print(f'P: {b.P}')
    
    for i in range(N_UPDATES):
        b.update()

    data = b.data
    print(data[-1])
    Visuals.create_visualization_grid(data, filename=f'{folder_path}/animation', duration=100, gif=True, video=False)
    Visuals.visualize_target_result(target, data, filename=f'{folder_path}/Result.png')
    Visuals.visualize_evolution_results(result_path=f'{folder_path}/evolution_rewards.json', filename=f'{folder_path}/Best_rewards.png')

