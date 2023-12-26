import numpy as np
from geneticalgorithm import geneticalgorithm as ga


def fitness_function(X):
    x = X[0]
    y = X[1]

    leather_used = x + y
    labor_used = x + 2 * y

    # Check constraints
    if leather_used > 40 or labor_used > 60 or x < 0 or y < 0:
        penalty = np.inf
    else:
        penalty = 0

    profit = 3 * x + 4 * y - penalty

    return -profit  # Minimize negative profit


algorithm_params = {
    'max_num_iteration': None,
    'population_size': 100,
    'mutation_probability': 0.4,  # Adjust as needed
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

# Convert variable_boundaries to a NumPy array
variable_boundaries = np.array([(0, 40), (0, 40)])

model = ga(
    function=fitness_function,
    dimension=2,
    variable_type='int',
    variable_boundaries=variable_boundaries,  # Use the NumPy array
    algorithm_parameters=algorithm_params
)

model.run()

# Display the results
best_solution = model.output_dict['variable']
max_profit = -model.output_dict['function']

print("Best Solution:", best_solution)
print("Max Profit:", max_profit)
