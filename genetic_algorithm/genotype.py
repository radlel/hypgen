# -*- coding: utf-8 -*-
"""

This module includes definition of class Genotype and its subclasses.

"""


from copy import copy, deepcopy
from math import floor
from pandas import DataFrame
from typing import List, Dict

from definitions.types import *
from definitions.config import *


class Gen:
    """ Definition of Gen - coordinates of single point """
    def __init__(self, plane: Plane) -> None:
        self.point = deepcopy(POINT_DEF[plane])


class Chromosome:
    """ Definition of Chromosome - collection of Gens """
    def __init__(self, plane: Plane) -> None:
        self.plane = plane
        self.gens = np.array([Gen(plane=plane) for _ in range(CHROMOSOME_SIZE)])
        self.gen_init_tangent = np.inf

    def init_tangent_rand(self) -> None:
        """
        Initializes tangent angle randomly
        Params:                                                                     type:
        @return: None
        """

        if self.plane == Plane.HORIZONTAL:
            """ Initialize tangent in first point (for horizontal drawn from range (ANGLE_MIN_H, ANGLE_MAX_H) """
            self.gen_init_tangent = (ANGLE_MAX_H - ANGLE_MIN_H) * np.random.sample() + ANGLE_MIN_H
        elif self.plane == Plane.VERTICAL:
            """ Initialize tangent in first point (for vertical drawn from range (ANGLE_MIN_V, ANGLE_MAX_V)) """
            self.gen_init_tangent = (ANGLE_MAX_V - ANGLE_MIN_V) * np.random.sample() + ANGLE_MIN_V
        else:
            raise ValueError('Invalid plane parameter value! {}'.format(self.plane))


class Genotype:
    """ Definition of Genotype - collection of Chromosomes """
    def __init__(self) -> None:
        self.chromosome_h = Chromosome(plane=Plane.HORIZONTAL)
        self.checksum_h = np.inf
        self.chromosome_v = Chromosome(plane=Plane.VERTICAL)
        self.checksum_v = np.inf

    def init_horizontal_random(self) -> None:
        """
        Initializes Gens (Points) for horizontal movement
        Params:                                                                     type:
        @return: None
        """

        """ Initialize start point and end point coordinates """
        gens = self.chromosome_h.gens
        gens[0].point['x'] = START_POINT['x']
        gens[0].point['y'] = START_POINT['y']
        gens[-1].point['x'] = END_POINT['x']
        gens[-1].point['y'] = END_POINT['y']

        """ Draw intermediate points coordinates """
        for gen_id in range(1, CHROMOSOME_SIZE - 1):
            while True:
                x_drawn = np.random.choice([i for i in range(DRAW_X_MIN, DRAW_X_MAX)], 1)[0]
                y_drawn = np.random.choice([i for i in range(DRAW_Y_MIN, DRAW_Y_MAX)], 1)[0]

                """ Check if selected point has proper z value (not np.inf) """
                if is_point_valid(x=x_drawn, y=y_drawn):
                    gens[gen_id].point['x'] = x_drawn
                    gens[gen_id].point['y'] = y_drawn
                    break
                else:
                    """" Drawn point not in range - draw another one """
                    pass

        """ For better convergence sort points to be from most far away to the closest end point """
        values_x = deepcopy([gen.point['x'] for gen in gens[1:-1]])
        values_y = deepcopy([gen.point['y'] for gen in gens[1:-1]])
        values_x.sort()
        values_y.sort()

        for gen_id in range(1, len(gens) - 1):
            (gens[gen_id]).point['x'] = values_x[gen_id - 1]
            (gens[gen_id]).point['y'] = values_y[gen_id - 1]

        """ Initialize tangent in first point """
        self.chromosome_h.init_tangent_rand()

        """ Calculate checksum based on points and tangent """
        self.init_checksum(plane=Plane.HORIZONTAL)

    def init_vertical_random(self) -> None:
        """
        Initializes Gens (Points) for vertical movement
        Params:                                                                     type:
        @return: None
        """

        """ Initialize start point and end point coordinates """
        gens = self.chromosome_v.gens
        gens[0].point['z'] = START_POINT['z']
        gens[0].point['d'] = 0
        gens[-1].point['z'] = END_POINT['z']
        gens[-1].point['d'] = ROUTE_RESOLUTION

        """ Draw intermediate points coordinates; z - height in adequate point d,
                                                  d - point in route len divided for 1000 equal segments """
        z_drawn = np.random.choice([i for i in range(ROUTE_MIN_HEIGHT, ROUTE_MAX_HEIGHT)], CHROMOSOME_SIZE - 2)

        """ Make sure d values are unique """
        while True:
            d_drawn = np.random.choice(ROUTE_RESOLUTION, CHROMOSOME_SIZE - 2)

            if len(d_drawn) == len(set(d_drawn)):
                """" d values are unique - ok """
                break
            else:
                """ d values are not unique, draw another collection """
                pass

        """ Sort values by growing distance d """
        df_vert = DataFrame({'z': z_drawn, 'd': d_drawn})
        df_vert = deepcopy(df_vert.sort_values(by=['d'], ignore_index=True))

        """ Set intermediate points """
        for gen_id in range(1, CHROMOSOME_SIZE - 1):
            gens[gen_id].point['z'] = df_vert['z'][gen_id - 1]
            gens[gen_id].point['d'] = df_vert['d'][gen_id - 1]

        """ Initialize tangent in first point """
        self.chromosome_v.init_tangent_rand()

        """ Calculate checksum based on points and tangent """
        self.init_checksum(plane=Plane.VERTICAL)

    def get_points_desc(self, plane: Plane) -> List[Dict[str, int]]:
        """
        Returns description dict for Gens representing points
        Params:                                                                     type:
        @param plane: Specifies vertical or horizontal plane                        Plane
        @return: Gens description                                                   List[Dict[str, int]]
        """

        if plane == Plane.HORIZONTAL:
            return [{'x': gen.point['x'], 'y': gen.point['y']} for gen in self.chromosome_h.gens]
        elif plane == Plane.VERTICAL:
            return [{'z': gen.point['z'], 'd': gen.point['d']} for gen in self.chromosome_v.gens]
        else:
            raise ValueError('Invalid plane parameter value! {}'.format(plane))

    def init_checksum(self, plane: Plane) -> int:
        """
        Calculates and returns Genotype checksum
        Params:                                                                     type:
        @param plane: Specifies vertical or horizontal plane                        Plane
        @return: checksum                                                           int
        """

        if plane == Plane.HORIZONTAL:
            self.checksum_h = (floor(sum([float(gen.point['x'] + gen.point['y']) for gen in
                               self.chromosome_h.gens[1:-1]]) + float(self.chromosome_h.gen_init_tangent) * 100))
            return copy(self.checksum_h)
        elif plane == Plane.VERTICAL:
            self.checksum_v = (floor(sum([float(gen.point['z'] + gen.point['d']) for gen in
                               self.chromosome_v.gens[1:-1]]) + float(self.chromosome_v.gen_init_tangent) * 100))
            return copy(self.checksum_v)


def is_point_valid(x: int, y: int) -> bool:
    """
    Checks if drawn coordinates are in map range and if corresponding z value in not np.inf
    Params:                                                                     type:
    @param x: Drawn x coordinate                                                int
    @param y: Drawn y coordinate                                                int
    @return: Information if drawn point is in map range                         bool
    """

    x_out, y_out = data.get_geo_data_out()
    if x in x_out and y in y_out:
        return False
    else:
        return True
