import logging
logging.basicConfig(level=logging.INFO)

import ConfigSpace.hyperparameters as CSH
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI
from smac.scenario.scenario import Scenario

import numpy as np
from genetic_algorithm import GeneticAlgorithm
from tsp import TspDrone
from parser import Parser


GA = GeneticAlgorithm()
TSPD = TspDrone

path = 'Raport/'

data_path = "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-12-n6.txt"
p = Parser(data_path)

points = np.array([p.X,p.Y])
points = points.T

truck_dist =[]
for el in points:
    d = np.linalg.norm(el - points, axis=1)
    truck_dist.append(d.tolist())
truck_dist = np.array(truck_dist)

drone_dist = []
for el in points:
    d = np.linalg.norm(el - points, axis=1) * p.speed_of_drone
    drone_dist.append(d.tolist())
drone_dist = np.array(drone_dist)

nr_params = points.shape[0]
nr_of_iteration = 200
nr_of_population = 50
GA.set_evaluation_function(TSPD(truck_dist, drone_dist))

def genetic_algorithm_test(params):
    mutation_prob = params["mutation_probability"]
    mutation_prob_drone = params["mutation_drone_probability"]
    chromosome_prob = params["chromosome_probability"]
    nr_of_population = params["poputation_size"]
    nr_of_iteration = params["nr_of_iteration"]

    
    GA.set_parameters(nr_params)
    GA.set_mutation_probability(mutation_prob)
    GA.set_mutation_drone_probability(mutation_prob_drone)
    GA.set_chromosome_probability(chromosome_prob)
    GA.set_population_size(nr_of_population)
    GA.set_iteration(nr_of_iteration)
    result, _, _ = GA.run(1)
    return min(result)



if __name__ == "__main__":
    cs = ConfigurationSpace()

    ## params
    p0 = CSH.UniformFloatHyperparameter("mutation_probability", 0.01, 0.09, default_value=0.05)
    p1 = CSH.UniformFloatHyperparameter("mutation_drone_probability", 0.01, 0.09, default_value=0.05)
    p2 = CSH.UniformFloatHyperparameter("chromosome_probability", 0.01, 0.09, default_value=0.05)
    p3 = CSH.UniformIntegerHyperparameter('poputation_size', lower=5, upper=150, log=False)
    p4 = CSH.UniformIntegerHyperparameter('nr_of_iteration', lower=2, upper=10, log=False)
    
    cs.add_hyperparameters([p0, p1, p2, p3, p4])

    # Scenario object
    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": 30,
                         "cs": cs,
                         "deterministic": "true"
                         })
    smac = SMAC4BB(scenario=scenario,
                   model_type='gp',
                   rng=np.random.RandomState(42),
                   acquisition_function=EI,
                   tae_runner=genetic_algorithm_test)

    smac.optimize()