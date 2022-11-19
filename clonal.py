import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class ClonalAlgorithym():
    
    def __init__(self):
        # Escolhendo uma "seed" para a geração de números
        np.random.seed(1)

    def generate_samples(self, N):
        return np.random.uniform(low=-10, high=10, size=(2,N)).transpose()
    
    def solve_bird_func(self,pair):
        # Aplica a "bird function"
        x,y = pair
        return np.sin(x)*np.exp(np.power(1-np.cos(y), 2))+np.cos(y)*np.exp(np.power(1-np.sin(x), 2))+np.power((x-y), 2)
    
    def solve_bird_func_for_stream(self, stream):
        # Aplica a "bird function" para um conjunto de pares
        return np.array([self.solve_bird_func(pair) for pair in stream])
    
    def linear_ranking(self, w, N):
        # Realizando o ranking linear, retornando um vetor que possui a aptidão de cada indivíduo em relação ao vetor de entrada
        result = np.array(self.solve_bird_func_for_stream(w))
        result_map = {}
        
        # Organiza as posições dos pares x e y de forma crescente de suas respostas
        for pos in range(N):
            result_map[pos] = result[pos]
        ranking_map = sorted(result_map.items(), key=lambda x:x[1])
        fitness_ranking = [rank for rank,value in ranking_map]  # Vetor que ordena os pares de mais apto para menos apto
        
        return fitness_ranking,result
        
        pair_rank = [(fitness_ranking[pos],pos) for pos in range(N)]
        pair_rank.sort()
        pair_rank = [N-1-rank for pos,rank in pair_rank] # Vetor onde cada posição se refere ao fitness do par que se encontra nesta posição
        
        return pair_rank

    def get_indexes(self,fitness,result,N):
       worst_index = fitness.index(min(fitness))
       worst_value = result[worst_index]
 
       best_index = fitness.index(max(fitness))
       best_value = result[best_index]
 
       avg_value = sum(result)/N
 
       return worst_value,avg_value,best_value
   
    def clone(self,group,cloning_rate):
       return(list(np.repeat(group, cloning_rate)))
    
    def mutate(self,pair):
        x,y = pair
        return x+random.uniform(-1,1),y+random.uniform(-1,1)
    
    def clone_and_mutate(self,population,size,fitness_ranking,cloning_rate,clone_amount,mutation_rate):
        fittest_group = fitness_ranking[0:clone_amount-1]
        clones = self.clone(fittest_group,cloning_rate)
        new_population_indexes = clones + [*range(size)]
        new_population = []
        
        mtt_const = (2*mutation_rate)/(size*(size+1))
        for pair in new_population_indexes:
            mtt_chance = mtt_const*(fitness_ranking.index(pair)+1)
            mtt_chance = 1 if(mtt_chance>1) else mtt_chance

            if(mtt_chance>=random.random()):
                new_population.append(self.mutate(population[pair]))
            else:
                new_population.append(population[pair])
        newest_population = np.array(new_population)
         
        return newest_population
    
    def select(self,population,limit):
        size = math.floor(population.size/2)
        fitness_ranking,result = self.linear_ranking(population,size)
        fittest_group = np.array([population[pair] for pair in fitness_ranking[0:limit]])

        return fittest_group
    
    def run(self,sample_size,n_iterations,cloning_rate,clone_amount,mutation_rate):
        N = sample_size
        w = self.generate_samples(N)
        worse_fit,best_fit,avg_fit = ([],[],[])
        populations = []

        for iteration in range(n_iterations):
            fitness_ranking,result = self.linear_ranking(w,N)
            worst_value,avg_value,best_value = self.get_indexes(fitness_ranking,result,N)
            adapted_population = self.clone_and_mutate(w,N,fitness_ranking,cloning_rate,clone_amount,mutation_rate)
            fittest_population = self.select(adapted_population,N)
            
        populations.append(fittest_population)
        return populations

    
if __name__ == "__main__":

    # Inicializando o algoritmo genético
    ca = ClonalAlgorithym()
    ca.run(sample_size=3,n_iterations=1,cloning_rate=2,clone_amount=2,mutation_rate=2)
