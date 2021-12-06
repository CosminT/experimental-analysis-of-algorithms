import numpy as np
from numpy.core.arrayprint import printoptions

class GeneticAlgorithm:
    def __init__(self):
        c = None
        self.parents = None
        self.population = None
        self.chromosome_size = None

        self.parameters = None
        self.mutation_prob = 0.5
        self.mutation_prob_drone = 0.5
        self.crossover_prob = 0.2
        self.chromosome_prob = 0.2
        self.population_size = 50
        
        self.max_fitness_values = []
        self.min_eval_values = []

    def mutate(self, x):
        if np.random.uniform(0, 1) < self.mutation_prob:
            el = np.random.randint(1, self.parameters)
            if np.random.uniform(0, 1) < self.mutation_prob_drone:
                x[1][el] = np.logical_not(x[1][el])
            else:
                x[0][el] = np.random.randint(1, self.parameters)
        return x

    def set_iteration(self, p):
        self.MAX = p

    def set_population_size(self, p):
        self.population_size = p
        self.population = np.array([self.generate() for _ in range(p)])

    def set_parameters(self, p):
        self.parameters = p
        self.chromosome_size = p * 2

    def generate(self):
        t = np.array([0])
        v = np.array([0])
        t = np.append(t, np.random.randint(1, self.parameters, size=(self.chromosome_size-1)))
        v = np.append(v, np.random.randint(2, size=(self.chromosome_size-1)))
        return np.array([t, v])

    def get_parrents(self, parents):
        result = []
        pairs_size = len(parents) / 2
        while len(result) < pairs_size:
            p_one = np.random.randint(len(parents))
            p_two = np.random.randint(len(parents))
            if p_one != p_two:
                result.append((parents[p_one],parents[p_two]))
                if p_one > p_two:
                    np.delete(parents, p_one)
                    np.delete(parents, p_two)
                else:
                    np.delete(parents, p_two)
                    np.delete(parents, p_one)          
        prob = np.random.uniform(0,1,len(result))
        result = np.array(result)
        return zip(result, prob)

    def cross_over(self, parents):
        children = np.zeros((self.population_size, 2, self.chromosome_size))
        parents = self.get_parrents(parents)
        i = 0 
        for pair, prob in parents:
            if prob > self.crossover_prob:
                crosspoint = np.random.randint(1, self.chromosome_size)
                children[i, 0:crosspoint] = pair[0][0:crosspoint]
                children[i, crosspoint:] = pair[1][crosspoint:]
                i += 1
                children[i, 0:crosspoint] = pair[1][0:crosspoint]
                children[i, crosspoint:] = pair[0][crosspoint:]
            else:
                children[i] = pair[0]
                i += 1
                children[i] = pair[1]
            i += 1
        return children

    # def test(self):
    #     print(self.population)
    #     print("-----")
    #     print(self.cross_over(self.population))

    def mutation(self, children):
        for c in range(self.population_size):
            if np.random.uniform(0,1) < self.chromosome_prob:
                children[c] = self.mutate(children[c])
        self.population = children

    def get_fitness(self):
        min_fitness = 1e9
        max_fitness = -1e9
        evaluated = np.zeros(self.population_size)
        for i in range(self.population_size):
            evaluated[i] = self.eval(self.population[i], self.parameters)
            if evaluated[i] < min_fitness:
                min_fitness = evaluated[i]
            if evaluated[i] > max_fitness:
                max_fitness = evaluated[i]

        fitness = 1.1 * max_fitness - evaluated
        self.min_eval_values.append(min(evaluated))
        self.max_fitness_values.append(max(fitness))
        return fitness

    def select(self, q, pick):
        for i in range(0, len(q) - 1):
            if q[i] <= pick < q[i + 1]:
                return i

    def selection(self, fitness):
        q = [0.0]; new_pop = []
        p = fitness / np.sum(fitness)
        for i in range(1, self.population_size + 1):
            q.append(q[i - 1] + p[i - 1])
        q.append(1.1)
        for i in range(self.population_size):
            pos = np.random.uniform(0, 1)
            new_pop.append(self.population[self.select(q, pos)])
        return np.array(new_pop)

    def set_fitness_function(self, eval):
        self.eval = eval

    def run(self, i):
        generation = 1
        while generation < self.MAX:
            fitness = self.get_fitness()
            parents = self.selection(fitness)
            children = self.cross_over(parents)
            self.mutation(children)
            if generation % 20 == 0:
                print('\x1b[6;30;42m' + str(i) + '\x1b[0m', generation, min(self.min_eval_values), min(self.max_fitness_values))
            generation += 1
        return self.min_eval_values, self.max_fitness_values