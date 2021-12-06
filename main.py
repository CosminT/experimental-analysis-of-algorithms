from parser import Parser
from genetic_algorithm import GeneticAlgorithm
import numpy as np

GA = GeneticAlgorithm()

p = Parser("data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/uniform-10-n5.txt",
"data/pcbouman-eur-TSP-D-Instances-8a6d795/uniform/solutions/uniform-10-n5-DP.txt")
p.get_input_data()

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

nr_params = drone_dist.shape[0]
nr_of_iteration = 1000
nr_of_population = 50


def check_cycle(v):
    result = [v[0]]
    for el in range(1,len(v)):
        if v[el-1] != v[el]:
            result.append(v[el])
    return not(len(np.unique(result)) == len(result))

def TSP(x, nr_of_nodes):
    x = x.astype(int)
    idx_dr = np.where(x[1] == 1)[0]
    idx_tr = np.where(x[1] == 0)[0]
    tour_dr = x[0][idx_dr]
    tour_tr = x[0][idx_tr]

    if np.array_equal(np.unique(x),np.arange(nr_of_nodes)) != True:
        return 1000

    if check_cycle(tour_tr):
        return 1000

    count = 0
    for el in np.unique(tour_dr):
        if el in np.unique(tour_tr):
            count+=1
    
    if count == 0:
        return 1000

    # print(x)
    # print(tour_dr)
    # print(tour_tr)
    # print("\n")

    # print(np.unique(tour_tr))
    # print(np.unique(tour_dr))

    tour_tr = np.append(tour_tr,[0])
    return truck_dist[tour_tr[:-1], tour_tr[1:]].sum() + drone_dist[tour_dr[:-1], tour_dr[1:]].sum()

GA.set_fitness_function(TSP)
GA.set_parameters(nr_params)
GA.set_population_size(nr_of_population)
GA.set_iteration(nr_of_iteration)
# GA.test()
GA.run(1)

# x = np.array([[0, 2, 2, 2, 4, 4, 2, 2, 3, 3], [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
# TSP(x, nr_params)