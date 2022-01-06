import numpy as np
from genetic_algorithm import GeneticAlgorithm
from tsp import TspDrone
from parser import Parser
import multiprocessing


GA = GeneticAlgorithm()
TSPD = TspDrone

path = 'Raport/'

data_paths = [
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-1-n5.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-12-n6.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-21-n7.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-31-n8.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-41-n9.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-51-n10.txt",
    "data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-61-n20.txt"
]

p = Parser(data_paths[0])

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
nr_of_experiments = 30

GA.set_evaluation_function(TSPD(truck_dist, drone_dist))
GA.set_parameters(nr_params)
GA.set_population_size(nr_of_population)
GA.set_iteration(nr_of_iteration)
# GA.run(1)
# GA.debug()


if __name__ == '__main__':

    experiments = list(range(nr_of_experiments))
    with multiprocessing.Pool(int(nr_of_experiments)) as p:
        GA.set_parameters(nr_params)
        GA.set_population_size(nr_of_population)
        GA.set_iteration(nr_of_iteration)
        result = p.map(GA.run, experiments)   
    min_eval_values, _ = zip(*result)
    np.savez(path + "p_" + str(nr_params) + "_evaluation_values", min_eval_values)