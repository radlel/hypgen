# -*- coding: utf-8 -*-
"""

This module includes definition of class Population.

"""


from typing import Union, List, Tuple
from copy import deepcopy
from math import floor, ceil
from pandas import DataFrame

from genetic_algorithm.individual import Individual
from definitions.types import *
from definitions.config import *


class Population:
    """ Definition of Population - collection of Genotypes """
    def __init__(self, pop_size=POPULATION_SIZE) -> None:
        self.individuals = np.array([Individual() for _ in range(pop_size)])

    def initialize_random(self, curr_pop_crcs: Union[Tuple[List, List], None] = None) -> None:
        """
        Initializes Individuals with random values
        Params:                                                                     type:
        @param curr_pop_crcs: existing Individuals checksums                        Union[Tuple[List, List], None]
        @return: None
        """

        print('Population: started random initialization')

        """ Get invalid coordinates so Genotype can check if drawn point is correct """

        for (individual, ind_id) in zip(self.individuals, range((len(self.individuals)))):
            print('Population: initialize Individual {}'.format((list(self.individuals)).index(individual)))
            while True:
                individual.initialize_random()

                """ Check if individual CRC is unique in context of this Population """
                if individual.genotype.checksum_h not in \
                        [ind.genotype.checksum_h for ind in self.individuals[:ind_id]] and \
                        individual.genotype.checksum_v not in \
                        [ind.genotype.checksum_v for ind in self.individuals[:ind_id]]:

                    """ If yes then check if is unique in context of super population - if exist """
                    if curr_pop_crcs is not None:
                        spr_pop_crcs_h, spr_pop_crcs_v = curr_pop_crcs
                        if individual.genotype.checksum_h not in spr_pop_crcs_h and \
                                individual.genotype.checksum_v not in spr_pop_crcs_v:
                            """ Individual is globally unique - init next individual """
                            break
                        else:
                            """ At least one of CRCs exists in super population, retry random init """
                            pass
                    else:
                        """ No super population - individual is unique """
                        break
                else:
                    """ At least one of CRCs exists in this population, retry random init """
                    pass

        print('Population: ended random initialization')

        print_population_info('Random initialization', pop=self.individuals)

    def create_new_generation(self) -> None:
        """
        Creates new set of Individuals
        Params:                                                                     type:
        @return: None
        """

        print('\n', 'Population: Start creation of new generation...')

        """ Create new offspring """
        offspring = np.array([Individual() for _ in range(len(self.individuals))])

        """ select 30% best parents """
        best_to_worst_parents = self.__get_best_individuals_sorted()

        print_population_info(title='Best parents', pop=best_to_worst_parents[:NEW_POP_BEST_PARENTS_NUM])

        offspring[NEW_POP_BEST_PARENTS_START:NEW_POP_BEST_PARENTS_NUM] =\
            deepcopy(best_to_worst_parents[0:NEW_POP_BEST_PARENTS_NUM])

        """ use best parents to cross over offspring and generate 50% of them """
        children = self.__cross_over_mutate(parents_sorted=best_to_worst_parents)

        print_population_info(title='Children after crossing over and mutation', pop=children)

        offspring[NEW_POP_CHILDREN_START:NEW_POP_CHILDREN_START + NEW_POP_CHILDREN_NUM] = deepcopy(children)

        """ create random 20% individuals """
        random_pop = Population(pop_size=NEW_POP_RANDOM_NUM)

        """ Get current offspring CRCs to make sure created randomly are not the same """
        offspring_crcs_h = [ind.genotype.checksum_h for ind in offspring]
        offspring_crcs_v = [ind.genotype.checksum_v for ind in offspring]

        random_pop.initialize_random(curr_pop_crcs=(offspring_crcs_h, offspring_crcs_v))

        offspring[NEW_POP_RANDOM_START:NEW_POP_RANDOM_START + NEW_POP_RANDOM_NUM] = deepcopy(random_pop.individuals)

        """ Set new population """
        for (individual, index) in zip(self.individuals, range(len(self.individuals))):
            individual.genotype = deepcopy(offspring[index].genotype)
            individual.fenotype = deepcopy(offspring[index].fenotype)

        print_population_info(title='New population', pop=self.individuals)

        print('\n', 'Population: ...Finished creation of new generation...')

    def __get_best_individuals_sorted(self) -> np.array:
        """
        Returns Individuals sorted by best fitness
        Params:                                                                     type:
        @return: Individuals sorted by best fitness                                 np.array
        """

        individuals = deepcopy(self.individuals)
        costs_desc = deepcopy([{'index': index, 'cost': individual.fenotype.fitness_val} for
                              (index, individual) in zip(range(len(individuals)), individuals)])

        sorted_indexes = [index for (cost, index) in sorted(zip([item['cost'] for item in costs_desc],
                                                                [item['index'] for item in costs_desc]))]

        return np.array([individuals[index] for index in sorted_indexes])

    def get_best_individual(self) -> Individual:
        """
        Returns an Individual with the best fitness
        Params:                                                                     type:
        @return: Individual with the best fitness                                   Individual
        """

        return (self.__get_best_individuals_sorted())[0]

    def __cross_over_mutate(self, parents_sorted: np.array) -> np.array:
        """
        Performs crossing over and mutation of Individuals in Population
        Params:                                                                     type:
        @param parents_sorted: Collection of the best Individuals                   np.array
        @return: Produced new Individuals                                           np.array
        """

        print('\t\t\tStarted crossing over and mutation...')

        """ Create new empty population - children """
        pop = Population(pop_size=NEW_POP_CHILDREN_NUM)
        children = pop.individuals
        parents = deepcopy(parents_sorted)
        child_id = 0

        """ Create list of mating ranges for every parent - every parent index from list parents indicates
            top value of selection range, bottom selection range is previous value in list """
        parents_mating_points = []
        for parent_id in range(len(parents)):
            parents_mating_points.append(ceil(parents[0].fenotype.fitness_val /
                                              parents[parent_id].fenotype.fitness_val *
                                              MATING_POINTS_MAX)
                                         + (parents_mating_points[parent_id - 1] if parent_id > 0 else 0))
        mating_points_sum = parents_mating_points[-1]

        """ Process crossing procedure till all children are produced """
        while True:
            """ Draw parents by 2 random points from range (0, mating_points_sum) """
            rand_mating_points = np.random.choice(mating_points_sum, 2)
            mating_point1 = rand_mating_points[0]
            mating_point2 = rand_mating_points[1]
            parent1, parent2 = None, None
            child = Individual()

            """ Evaluate parents """
            for mate_range, parent in zip(parents_mating_points, parents):
                if mating_point1 < mate_range and parent1 is None:
                    parent1 = parent

                if mating_point2 < mate_range and parent2 is None:
                    parent2 = parent

                """ Check if parents already chosen """
                if parent1 is not None and parent2 is not None:
                    break

            """ Check if chosen parents are not the same Individual """
            if ((parent1.genotype.checksum_h != parent2.genotype.checksum_h) and
                    (parent1.genotype.checksum_v != parent2.genotype.checksum_v)):

                """ Generate crossing masks and make sure it is not uniform """
                while True:
                    cross_mask_h = np.random.choice([0, 1], CHROMOSOME_SIZE - 2)
                    if not sum(cross_mask_h) in [0, len(cross_mask_h)]:
                        break

                while True:
                    cross_mask_v = np.random.choice([0, 1], CHROMOSOME_SIZE - 2)
                    if not sum(cross_mask_v) in [0, len(cross_mask_v)]:
                        break

                """ Generate mask for choosing init tangent in starting point"""
                tang_mask_h = np.random.choice([0, 1], 1)[0]
                tang_mask_v = np.random.choice([0, 1], 1)[0]

                """ Start and end point is always the same """
                child.genotype.chromosome_h.gens[0] = deepcopy(parent1.genotype.chromosome_h.gens[0])
                child.genotype.chromosome_h.gens[-1] = deepcopy(parent1.genotype.chromosome_h.gens[-1])
                child.genotype.chromosome_v.gens[0] = deepcopy(parent1.genotype.chromosome_v.gens[0])
                child.genotype.chromosome_v.gens[-1] = deepcopy(parent1.genotype.chromosome_v.gens[-1])

                """ Copy intermediate gens according to crossing mask """
                for i in range(1, CHROMOSOME_SIZE - 1):
                    child.genotype.chromosome_h.gens[i] = \
                        deepcopy((parent1.genotype.chromosome_h.gens[i]) if (cross_mask_h[i - 1] == 0)
                                 else (parent2.genotype.chromosome_h.gens[i]))

                    child.genotype.chromosome_v.gens[i] = \
                        deepcopy((parent1.genotype.chromosome_v.gens[i]) if (cross_mask_v[i - 1] == 0)
                                 else (parent2.genotype.chromosome_v.gens[i]))

                """ Vertical points must be ascending - sort them if they are not """
                z_vals = [gen.point['z'] for gen in child.genotype.chromosome_v.gens[1:-1]]
                d_vals = [gen.point['d'] for gen in child.genotype.chromosome_v.gens[1:-1]]

                """ d values must be unique """
                if len(d_vals) == len(set(d_vals)):
                    """" d values are unique - ok """

                    df_vert = DataFrame({'z': z_vals, 'd': d_vals})
                    df_vert = deepcopy(df_vert.sort_values(by=['d'], ignore_index=True))
                    """ Set intermediate points """
                    for gen_id in range(1, CHROMOSOME_SIZE - 1):
                        child.genotype.chromosome_v.gens[gen_id].point['z'] = deepcopy(df_vert['z'][gen_id - 1])
                        child.genotype.chromosome_v.gens[gen_id].point['d'] = deepcopy(df_vert['d'][gen_id - 1])

                    child.genotype.chromosome_h.gen_init_tangent = deepcopy(
                        parent1.genotype.chromosome_h.gen_init_tangent if (tang_mask_h == 0)
                        else parent2.genotype.chromosome_h.gen_init_tangent)

                    child.genotype.chromosome_v.gen_init_tangent = deepcopy(
                        parent1.genotype.chromosome_v.gen_init_tangent if (tang_mask_v == 0)
                        else parent2.genotype.chromosome_v.gen_init_tangent)

                    """ Process mutation """
                    child = self.__mutate(child=child)

                    """ Check if created individual does not already exist """
                    crc_h_child = child.genotype.init_checksum(plane=Plane.HORIZONTAL)
                    crc_v_child = child.genotype.init_checksum(plane=Plane.VERTICAL)

                    if ((crc_h_child not in [ind.genotype.checksum_h for ind in parents] and
                         (crc_v_child not in [ind.genotype.checksum_v for ind in parents])) and
                            (crc_h_child not in [ind.genotype.checksum_h for ind in children[:child_id]]) and
                            (crc_v_child not in [ind.genotype.checksum_v for ind in children[:child_id]])):

                        """ Initialize individual fenotype and check if is valid """
                        individual_valid = child.init_fenotype_is_valid()
                        if individual_valid is True:

                            """ Add individual to new population """
                            children[child_id] = deepcopy(child)
                            print('\t\t\t\tChild crossing/mutation successful:', child_id)
                            child_id += 1

                            if child_id == NEW_POP_CHILDREN_NUM:
                                """ Produced all children, return """
                                print('\t\t\t...Ended crossing over and mutation')
                                return children

                            else:
                                """ Not all children produced yet - go to beginning of the loop """
                                pass
                        else:
                            """ Individual invalid - go to beginning of the loop """
                            pass
                    else:
                        """ Individual already exists in population - go to beginning of the loop """
                        pass
                else:
                    """ Invalid d values collection, try again """
                    pass
            else:
                """ Invalid parents drawing - go to beginning of the loop """
                pass

    def __mutate(self, child: Individual) -> Individual:
        """
        Performs Individuals mutation
        Params:                                                                     type:
        @param child: Child Individual                                              Individual
        @return: Mutated Individual                                                 Individual
        """

        """ Horizontal mutation """
        mutate_mask = np.random.choice(10, 1)[0]

        """ Drawing 0 is 10% chance - mutations probability """
        if mutate_mask == 0:
            """ Draw which gen will be affected """

            mut_gen_id = (np.random.choice(CHROMOSOME_SIZE - 2, 1))[0] + 1

            """ Draw new point and replace """
            x_drawn = np.random.choice([i for i in range(DRAW_X_MIN, DRAW_X_MAX)], 1)[0]
            y_drawn = np.random.choice([i for i in range(DRAW_Y_MIN, DRAW_Y_MAX)], 1)[0]

            child.genotype.chromosome_h.gens[mut_gen_id].point['x'] = x_drawn
            child.genotype.chromosome_h.gens[mut_gen_id].point['y'] = y_drawn
        else:
            """ Children is not about to be mutated in horizontal, just pass """
            pass

        """ Vertical mutation """
        mutate_mask = np.random.choice(10, 1)[0]
        """ Drawing 0 is 10% chance - mutations probability """
        if mutate_mask == 0:
            """ Draw which gen will be affected """

            mut_gen_id = (np.random.choice(CHROMOSOME_SIZE - 2, 1))[0] + 1

            """ Draw new z value and replace """
            z_drawn = np.random.choice([i for i in range(ROUTE_MIN_HEIGHT, ROUTE_MAX_HEIGHT)], 1)[0]

            child.genotype.chromosome_v.gens[mut_gen_id].point['z'] = z_drawn
        else:
            """ Children is not about to be mutated in vertical, just pass """
            pass

        """ Tangent angle h mutation """
        mutate_mask = np.random.choice(10, 1)[0]
        """ Drawing 0 is 10% chance - mutations probability """
        if mutate_mask == 0:
            """ Draw new angle value and replace """
            child.genotype.chromosome_h.init_tangent_rand()
        else:
            """ Children is not about to be mutated in angle h, just pass """
            pass

        """ Tangent angle v mutation """
        mutate_mask = np.random.choice(10, 1)[0]
        """ Drawing 0 is 10% chance - mutations probability """
        if mutate_mask == 0:

            """ Draw new angle value and replace """
            child.genotype.chromosome_v.init_tangent_rand()
        else:
            """ Children is not about to be mutated in angle v, just pass """
            pass

        return child


def print_population_info(title: str, pop: np.array) -> None:
    """
    Prints current Population details
    Params:                                                                     type:
    @param title: Context info                                                  str
    @param pop: Collection of Individuals                                       np.array
    @return: None
    """

    pop = deepcopy(pop)
    gens_desc_h = [[(floor(gen.point['x']), floor(gen.point['y'])) for
                    gen in ind.genotype.chromosome_h.gens] for ind in pop]
    gens_desc_v = [[(floor(gen.point['d']), floor(gen.point['z'])) for
                    gen in ind.genotype.chromosome_v.gens] for ind in pop]

    crcs_h = [ind.genotype.checksum_h for ind in pop]
    crcs_v = [ind.genotype.checksum_v for ind in pop]

    tangs_h = [round(float(ind.genotype.chromosome_h.gen_init_tangent), 2) for ind in pop]
    tangs_v = [round(float(ind.genotype.chromosome_v.gen_init_tangent), 2) for ind in pop]

    i = 0
    print('Population info:', title)
    for index in range(len(pop)):
        print('\t{}\tH:'.format(i), gens_desc_h[index], '[{:4.2f}]'.format(tangs_h[index]), 'CRC:', crcs_h[index])
        print('\t\tV:', gens_desc_v[index], '[{:4.2f}]'.format(tangs_v[index]), 'CRC:', crcs_v[index])
        print('\t\tF:', format_fitness(fitness=pop[index].fenotype.fitness_val))
        i += 1


def format_fitness(fitness: float) -> str:
    """
    Formats fitness for printing needs
    Params:                                                                     type:
    @param fitness: Individual fitness value                                    float
    @return: Fitness formatted to string                                        str
    """

    fitness_str = str(floor(fitness))
    strl = len(fitness_str)
    n = floor(strl / 3)

    if n > 0 and strl > 3:
        r = strl % 3
        if r == 0:
            n_dots = n - 1
        else:
            n_dots = n

        for i in range(n_dots):
            fitness_str = fitness_str[:-3 * (i + 1) - i] + "'" + fitness_str[-3 * (i + 1) - i:]

    return fitness_str
