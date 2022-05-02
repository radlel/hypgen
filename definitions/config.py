# -*- coding: utf-8 -*-
"""

This file includes configuration parameters for genetic algorithm.


"""


from data.data_man import GeoData


""" Genetic Algorithm parameters """
CHROMOSOME_SIZE = 6
GENERATIONS_NUM = 100

NEW_POP_BEST_PARENTS_NUM = 30
NEW_POP_CHILDREN_NUM = 50
NEW_POP_RANDOM_NUM = 20

POPULATION_SIZE = NEW_POP_BEST_PARENTS_NUM + NEW_POP_CHILDREN_NUM + NEW_POP_RANDOM_NUM

NEW_POP_BEST_PARENTS_START = 0
NEW_POP_CHILDREN_START = NEW_POP_BEST_PARENTS_NUM
NEW_POP_RANDOM_START = NEW_POP_BEST_PARENTS_NUM + NEW_POP_CHILDREN_NUM

ROUTE_RESOLUTION = 1000
MATING_POINTS_MAX = 100

""" Container for geo data """
data = GeoData()
