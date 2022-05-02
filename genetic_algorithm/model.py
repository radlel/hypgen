# -*- coding: utf-8 -*-
"""

This module includes definition of class GAModel which stands for
Genetic Algorithm Model.

"""


import matplotlib.pyplot as plt
import os
from copy import deepcopy

from genetic_algorithm.population import Population, format_fitness
from geometry.route_calc import plot_route_2d
from definitions.types import *
from definitions.config import *


class GAModel:
    """ Definition of GAModel - genetic algorithm manager """
    def __init__(self) -> None:
        create_output_dir()
        self.population = Population(pop_size=POPULATION_SIZE)
        self.population.initialize_random()

    def evaluate(self) -> None:
        """
        Performs consecutive algorithm iterations
        Params:                                                                     type:
        @return:
        """

        best_ind = None
        fitness_hist = []

        """ Perform algorithm iterations """
        for i in range(GENERATIONS_NUM):
            print('Model: Create new generation:', i)
            self.population.create_new_generation()

            best_ind = deepcopy(self.population.get_best_individual())
            fitness = best_ind.fenotype.fitness_val
            print('\t^ Best fitness:', format_fitness(fitness=fitness), '\n')
            fitness_hist.append(fitness)

            plt.plot(range(len(fitness_hist)), fitness_hist)
            plt.title('fitness: {}'.format(i))
            plt.grid()
            plt.xlabel('Iteration no')
            plt.ylabel('Cost [mln euro]')
            plt.savefig('output\\' + 'fitness_' + str(i) + '.png')
            plt.close()

        """ After algorithm is done plot best route """
        best_route_desc_h = best_ind.fenotype.get_route_desc(plane=Plane.HORIZONTAL)
        best_route_len_h = best_ind.fenotype.route_len_h
        best_p_desc_h = best_ind.genotype.get_points_desc(plane=Plane.HORIZONTAL)
        plot_route_2d(plane=Plane.HORIZONTAL, route_desc=best_route_desc_h, route_len=best_route_len_h,
                      p_dicts=best_p_desc_h, title='end best H')

        discretized_points = best_ind.discretize_route()
        landform_heights = best_ind.get_landform_route_heights(route_points=discretized_points)
        d_points_disc = [elem['d'] for elem in discretized_points]
        landform = (d_points_disc, landform_heights)

        best_route_desc_v = best_ind.fenotype.get_route_desc(plane=Plane.VERTICAL)
        best_route_len_v = best_ind.fenotype.route_len_total
        plot_route_2d(plane=Plane.VERTICAL, route_desc=best_route_desc_v, route_len=best_route_len_v,
                      p_dicts=best_ind.map_genotype_v_to_p_dicts(), landform=landform, title='end best V',
                      z_min=min(landform_heights), z_max=max(landform_heights))

        plt.plot(range(len(fitness_hist)), fitness_hist)
        plt.title('fitness history')
        plt.grid()
        plt.xlabel('Iteration no')
        plt.ylabel('Cost [mln euro]')
        plt.savefig('output\\' + 'fitness_change' + '.png')
        plt.close()


def create_output_dir() -> None:
    """
    Checks if output directory exists and if not creates it
    Params:                                                                     type:
    @return: None
    """

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    else:
        """ Directory already exists """
        pass
