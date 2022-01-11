import numpy as np

class GeneticAlgorithm:
    def __init__(self):
        c = None
        self.parents = None
        self.population = None
        self.chromosome_size = None

        self.parameters = None
        self.mutation_prob = 0.02
        self.mutation_prob_drone = 0.05
        self.crossover_prob = 0.01
        self.chromosome_prob = 0.02
        self.population_size = 50
        
        self.max_fitness_values = []
        self.min_eval_values = []

        self.posibil = [[0,0],[1,0],[0,1]]
        self.BEST_TOUR = []

    def mutate(self, x):
        if np.random.uniform(0, 1) < self.mutation_prob:
            if np.random.uniform(0, 1) < self.mutation_prob_drone:
                p = np.random.randint(0, int(self.parameters/2))
                result = x[1,1:-(1+self.parameters%2)]
                # print(data, self.parameters%2)
                result = result.reshape(int(self.parameters/2),2)
                # print(p)
                if p == 0:
                    if sum(result[p+1]) == 0:
                        result[p] = self.posibil[1]
                    else:
                        result[p] = self.posibil[np.random.randint(1,3)]

                elif p == len(result) - 1:
                    if sum(result[p-1]) == 0:
                        result[p] = self.posibil[1]
                    else:
                        result[p] = self.posibil[np.random.randint(0,3)]
                else:
                    if sum(result[p-1]) == 0 and sum(result[p+1]) == 0:
                        result[p] = self.posibil[np.random.randint(1,2)]
                    elif sum(result[p-1]) == 0 and sum(result[p+1]) == 1 and result[p+1,1] == 0 :
                        result[p] = self.posibil[1]
                    elif sum(result[p+1]) == 0 and sum(result[p-1]) == 1 and result[p-1,1] == 0 :
                        result[p] = self.posibil[2]
                    else:
                        result[p] = self.posibil[np.random.randint(0,3)]
                result = result.reshape(-1)
                x[1,1:-(1+self.parameters%2)] = result

            else:
                a = np.random.randint(1, self.chromosome_size-1)
                b = np.random.randint(1, self.chromosome_size-1)
                tmp = x[0][a]
                x[0][a] = x[0][b]
                x[0][b] = tmp
        return x

    def set_iteration(self, p):
        self.MAX = p

    def set_mutation_probability(self, p):
        self.mutation_prob = p

    def set_mutation_drone_probability(self, p):
        self.mutation_prob_drone = p

    def set_chromosome_probability(self, p):
        self.chromosome_prob = p

    def set_population_size(self, p):
        self.population_size = p
        self.population = np.array([self.generate() for _ in range(p)])

    def set_parameters(self, p):
        self.parameters = p - 1
        self.chromosome_size = p + 1

    def generate_bits(self):
        posibil = [[0,0],[1,0],[0,1]]
        result = []
        for _ in range(int(self.parameters/2)):
            if len(result) == 0:
                result += posibil[np.random.randint(0,3)]
            else:
                if sum(result[-2:]) == 0:
                    result += posibil[1]
                elif result[-1] == 1:
                    tp = np.random.randint(0,2)
                    if tp == 1:
                        result += posibil[2]
                    else:
                        result += posibil[0]
                else:
                    result += posibil[1]
        if self.parameters%2 == 1:
            result.append(0)
        result = [1] + result
        result.append(1)
        return result

    def generate(self):
        cities = np.array([0])
        el = np.arange(1,self.parameters+1)
        np.random.shuffle(el)
        cities = np.append(cities, el)
        cities = np.append(cities,[0])
        bits = self.generate_bits()
        return np.array([cities, bits])

    def get_parrents(self, parents):
        result = []
        pairs_size = len(parents) / 2
        while len(result) < pairs_size:
            p_one = np.random.randint(len(parents))
            p_two = np.random.randint(len(parents))
            if p_one != p_two:
                result.append([parents[p_one],parents[p_two]])
                if p_one > p_two:
                    np.delete(parents, p_one)
                    np.delete(parents, p_two)
                else:
                    np.delete(parents, p_two)
                    np.delete(parents, p_one)          
        prob = np.random.uniform(0,1,len(result))
        result = np.array(result)
        # print(result.shape)
        return zip(result, prob)

    def cross_over(self, parents):
        children = []
        parents = self.get_parrents(parents)
        for pair, prob in parents:
            if prob > self.crossover_prob:
                crosspoint = np.random.randint(1, self.chromosome_size-3)
                aux = pair[:,:,1:pair.shape[2]-1]
                param = pair.shape[2]-2
                result = []
                for el in aux[:,0]:
                    tmp = np.arange(1,param+1)
                    r = []
                    for i in el:
                        idx = tmp.tolist().index(i)
                        tmp = np.delete(tmp, idx)
                        r.append(idx+1)
                    result.append(r)

                cross = [result[0][:crosspoint] + result[1][crosspoint:], result[1][:crosspoint] + result[0][crosspoint:]]

                output = []
                for el in cross:
                    tmp = np.arange(1,param+1)
                    r = []
                    for i in el:
                        idx = tmp[i-1]
                        tmp = np.delete(tmp, i-1)
                        r.append(idx)
                    output.append(r)

                pair[:,0,1:pair.shape[2]-1] = output
                for p in pair:
                    children.append(p)
            else:
                for p in pair:
                    children.append(p)
        return np.array(children)

    def debug(self):
        ## debug function
        print(self.generate())
        # print(self.mutate(self.population[0]))
        # print(self.population.shape)
        # print(self.population[0].shape)
        # print(self.mutate(self.population[0]))
        # print(self.cross_over(self.population).shape)

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
            evaluated[i] = self.eval.get_score(self.population[i])
            if evaluated[i] < min_fitness:
                min_fitness = evaluated[i]
            if evaluated[i] > max_fitness:
                max_fitness = evaluated[i]

        fitness = 1.1 * max_fitness - evaluated
        self.BEST_TOUR = self.population[np.argmin(evaluated)]
        # print(evaluated[np.argmin(evaluated)])
        self.min_eval_values.append(evaluated[np.argmin(evaluated)])
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

    def set_evaluation_function(self, eval):
        self.eval = eval

    def run(self, i):
        generation = 1
        while generation < self.MAX:
            fitness = self.get_fitness()
            parents = self.selection(fitness)
            children = self.cross_over(parents)
            self.mutation(children)
            if generation % 20 == 0:
                print('\x1b[6;30;42m' + str(i) + '\x1b[0m', generation, min(self.min_eval_values), "Fitness:", min(self.max_fitness_values))
            generation += 1 
        print(self.BEST_TOUR)
        print(min(self.min_eval_values))
        return self.min_eval_values, self.max_fitness_values, self.BEST_TOUR