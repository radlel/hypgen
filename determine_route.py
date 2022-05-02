# -*- coding: utf-8 -*-
"""

This application generates optimal route for Hyperloop infrastructure between
given points A and B with consideration of landform, cost, infrastructure type,
route limitations and more. Due to high complexity level and great number of
factors artificial intelligence algorithm is used.

In this module the instance of GAModel (Genetic Algorithm Model) is created.
Based on provided input data, model finds eligible solution with the best
characteristics.

Example:
    Default configuration for point A is Krakow and for point B is Warsaw.
    This can be changed by adjusting parameters in modules config.py and types.py
    and providing geological data in resources\geo_data.

Execution:
    $ python determine_route.py

"""

__author__ = 'Radoslaw Lelito'
__email__ = 'radoslaw.lelito@gmail.com'
__date__ = '2021/01/21'
__license__ = 'MIT'
__version__ = '2.0.0'


from genetic_algorithm.model import GAModel


if '__main__' == __name__:
    model = GAModel()
    model.evaluate()
