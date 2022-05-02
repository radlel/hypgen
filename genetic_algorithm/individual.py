# -*- coding: utf-8 -*-
"""

This module includes definition of class Individual which represents
single unit in genetic algorithm population.

"""


import sympy as sp
import logging
from typing import List, Dict, Union, Tuple, Any
from math import floor, ceil
from copy import deepcopy, copy

from genetic_algorithm.genotype import Genotype, is_point_valid
from genetic_algorithm.fenotype import Fenotype
from geometry.route_gen import get_route_description, dir_to_rad
from geometry.route_calc import plot_route_2d
from data.data_man import get_geo_data_element_z
from definitions.types import *
from definitions.config import *


class Individual:
    """ Definition of Individual - owner of Genotype and Fenotype """

    individual_unique_id = 0

    def __init__(self) -> None:
        logging.basicConfig(filename=OUTPUT_PATH + 'results.log', filemode='w', format='%(message)s',
                            level=logging.INFO)

        self.genotype = Genotype()
        self.fenotype = Fenotype()
        self.discrete_route_h_points = None

    def init_fenotype_is_valid(self) -> bool:
        """
        Initializes Fenotype and checks for its validity
        Params:                                                                     type:
        @return: Information if Fenotype is valid                                   bool
        """

        self.fenotype.init_horizontal(p_dicts=self.genotype.get_points_desc(plane=Plane.HORIZONTAL),
                                      init_tangent=self.genotype.chromosome_h.gen_init_tangent)
        disc_route_ret = self.discretize_route()

        """ Check if all discrete route points are in known region """
        if disc_route_ret is not None:
            """ Route points are in known region - ok """
            self.discrete_route_h_points = disc_route_ret
        else:
            """ Route points not in region - return False """
            return False

        self.fenotype.init_vertical(p_dicts=self.map_genotype_v_to_p_dicts(),
                                    init_tangent=self.genotype.chromosome_v.gen_init_tangent)

        for arc_desc in self.fenotype.route_desc_v:
            if arc_desc['arc_rad_len'] > np.pi:
                self.discrete_route_h_points = None
                return False

        """ Fenotype is ok, calculate fitness """
        self.calculate_fitness()

        self.discrete_route_h_points = None
        return True

    def initialize_random(self) -> None:
        """
        Initializes Genotype randomly until is valid
        Params:                                                                     type:
        @return: None
        """

        while True:
            self.genotype.init_horizontal_random()
            self.genotype.init_vertical_random()
            individual_valid = self.init_fenotype_is_valid()

            if individual_valid is True:
                break

    def map_genotype_v_to_p_dicts(self) -> List[Dict[str, int]]:
        """
        Converts vertical points from Genotype to description dict
        Params:                                                                     type:
        @return: Description dict                                                   List[Dict[str, int]]
        """

        points_v = deepcopy(self.genotype.get_points_desc(plane=Plane.VERTICAL))
        route_len_h = copy(self.fenotype.route_len_h)
        return [{'z': point['z'], 'd': floor(point['d'] / ROUTE_RESOLUTION * route_len_h)} for point in points_v]

    def calculate_fitness(self) -> None:
        """
        Calculated fitness of an Individual
        Params:                                                                     type:
        @return: None
        """

        """ Take discretized route """
        route_points = self.discrete_route_h_points

        """ Get vector of height points from generated route """
        gen_z_vals = self.get_gen_route_heights(route_points=route_points, route_v_desc=self.fenotype.route_desc_v)

        """ Get vector of height points from landform """
        orig_landform_z_vals = self.get_landform_route_heights(route_points=route_points)

        if len(gen_z_vals) != len(orig_landform_z_vals):
            raise ValueError('Len of arrays must be equal!')

        """ Create vector of height differences """
        diff_vector_z = [gen_z_vals[i] - orig_landform_z_vals[i] for i in range(len(gen_z_vals))]

        cost, summary = self.calculate_route_cost(diff_vector_z=diff_vector_z)

        """ Check if arc is not to tight, in case it is add penalty """
        radius_lens = [int(arc['circle_radius_len']) for arc in self.fenotype.route_desc_h]
        for radius_len in radius_lens:
            if radius_len < 23000:
                cost += PENALTY_TIGHT_ARC
                break

        self.fenotype.fitness_val = cost

        """ Save plot representing generated route """
        if PLOT_SAVE:
            plot_route_2d(plane=Plane.HORIZONTAL,
                          route_desc=self.fenotype.route_desc_h,
                          route_len=self.fenotype.route_len_h,
                          p_dicts=self.genotype.get_points_desc(plane=Plane.HORIZONTAL),
                          title=str(Individual.individual_unique_id) + 'h')

            p_dicts = self.map_genotype_v_to_p_dicts()
            route_desc, route_len = get_route_description(plane=Plane.VERTICAL,
                                                          p_dicts=p_dicts,
                                                          init_tangent=self.genotype.chromosome_v.gen_init_tangent)

            d_points_disc = [elem['d'] for elem in route_points]
            landform = (d_points_disc, orig_landform_z_vals)
            z_min = min(gen_z_vals)
            z_max = max(gen_z_vals)
            plot_route_2d(plane=Plane.VERTICAL,
                          route_desc=route_desc,
                          route_len=route_len,
                          p_dicts=p_dicts,
                          title=str(Individual.individual_unique_id) + 'v',
                          landform=landform,
                          z_min=z_min,
                          z_max=z_max)

            logging.info(str(Individual.individual_unique_id) + ' ' +
                         str(int(cost)) + ' ' +
                         str(summary) + ' ' +
                         str(self.fenotype.route_desc_h) + ' ' +
                         str(self.fenotype.route_desc_v))
            Individual.individual_unique_id += 1

    def calculate_route_cost(self, diff_vector_z: List[float]) -> (int, Dict):
        """
        Calculates route cost
        Params:                                                                     type:
        @param diff_vector_z: Height differences on z axis                          List[float]
        @return: Cost and number of kilometers of every type of infrastructure      Tuple[int, Dict[str, Any]]
        """

        cost = 0
        tu, ex, gr, em, py = 0, 0, 0, 0, 0,

        for diff in diff_vector_z:

            """ Check if point is valid and has no np.inf """
            if diff == -np.inf:
                return MAX_COST, {'tu': 0, 'ex': 0, 'gr': 0, 'em': 0, 'py': 0}

            try:
                if diff <= TERRAIN_TUNNEL_HIGH:
                    """ Tunnel """
                    cost += COST_TUBE + COST_TUNNEL_BASE * (2 ** floor(abs(diff) / 50))
                    tu += 1
                elif TERRAIN_EXCAVATION_LOW < diff <= TERRAIN_EXCAVATION_HIGH:
                    """ Excavation """
                    cost += COST_TUBE + COST_EXC
                    ex += 1
                elif TERRAIN_GROUND_LOW < diff <= TERRAIN_GROUND_HIGH:
                    """ On ground """
                    cost += COST_TUBE
                    gr += 1
                elif TERRAIN_EMBANKMENT_LOW < diff <= TERRAIN_EMBANKMENT_HIGH:
                    """ Embankment """
                    cost += COST_TUBE + COST_EMB
                    em += 1
                elif diff > TERRAIN_PYLON_LOW:
                    """ Pylon """
                    cost += COST_TUBE + COST_PYLON_PARAM * (diff ** 2)
                    py += 1
                else:
                    raise ValueError

                """ Check if cost is not about to limit """
                if cost >= MAX_COST:
                    tu, ex, gr, em, py = 0, 0, 0, 0, 0,
                    return MAX_COST, {'tu': tu, 'ex': ex, 'gr': gr, 'em': em, 'py': py}

            except OverflowError:
                return MAX_COST, {'tu': 0, 'ex': 0, 'gr': 0, 'em': 0, 'py': 0}

        """ Add maintenance costs in 10 years """
        cost *= 2

        return cost, {'tu': tu, 'ex': ex, 'gr': gr, 'em': em, 'py': py}

    def is_route_in_region(self, route_points: List[Dict[str, Union[int, float]]]) -> bool:
        """
        Checks if route goes through considered region
        Params:                                                                     type:
        @param route_points: Consecutive route points                               List[Dict[str, Union[int, float]]]
        @return: Information if route goes through considered region                bool
        """

        x_collection = [elem['x'] for elem in route_points]
        y_collection = [elem['y'] for elem in route_points]
        x_min = min(x_collection)
        x_max = max(x_collection)
        y_min = min(y_collection)
        y_max = max(y_collection)

        if (x_min >= X_MIN_ALLOWED and x_max <= X_MAX_ALLOWED and y_min >= Y_MIN_ALLOWED and y_max <= Y_MAX_ALLOWED and
            is_point_valid(x=x_min, y=y_min) and is_point_valid(x=x_min, y=y_max) and
            is_point_valid(x=x_max, y=y_max) and is_point_valid(x=x_max, y=y_min)):
            return True
        else:
            return False

    def get_landform_route_heights(self, route_points: List[Dict[str, Union[int, float]]]) -> List[float]:
        """
        Returns landform line
        Params:                                                                     type:
        @param route_points: Consecutive route points                               List[Dict[str, Union[int, float]]]
        @return: List representing landform                                         List[float]
        """

        z_orig_vals = []

        for point in route_points:
            x, y = round(point['x'], -3), round(point['y'], -3)

            if x == X_MIN_ALLOWED - DIST1KM:
                x = X_MIN_ALLOWED
            elif x == X_MAX_ALLOWED + DIST1KM:
                x = X_MAX_ALLOWED

            if y == Y_MIN_ALLOWED - DIST1KM:
                y = Y_MIN_ALLOWED
            elif y == Y_MAX_ALLOWED + DIST1KM:
                y = Y_MAX_ALLOWED

            geo_data_x, geo_data_y, geo_data_z = data.get_geo_data_all()

            z_orig_vals.append(get_geo_data_element_z(z_data=geo_data_z,
                                                      x_coordinate=x,
                                                      y_coordinate=y,
                                                      x_range=geo_data_x,
                                                      y_range=geo_data_y))

        return z_orig_vals

    def get_gen_route_heights(self, route_points: List[Dict[str, Union[int, float]]],
                              route_v_desc: List[Dict[str, Any]]) -> List[float]:
        """
        Generates discrete representation of generated route heights
        Params:                                                                     type:
        @param route_points: Consecutive route points                               List[Dict[str, Union[int, float]]]
        @param route_v_desc: Vertical route description                             List[Dict[str, Any]]
        @return: Route heights                                                      List[float]
        """

        """ Evaluate common points of fallowing arcs """
        d_points_mapped = [d_elem['d'] for d_elem in self.map_genotype_v_to_p_dicts()]

        generated_route_z_values = []

        for d_point in [floor(d_elem['d']) for d_elem in route_points]:
            if d_points_mapped[0] <= d_point < d_points_mapped[1]:
                """ Take first arc for calculations """
                arc_id = 0
                pass
            elif d_points_mapped[1] <= d_point < d_points_mapped[2]:
                """ Take second arc for calculations """
                arc_id = 1
            elif d_points_mapped[2] <= d_point < d_points_mapped[3]:
                """ Take third arc for calculations """
                arc_id = 2
            elif d_points_mapped[3] <= d_point <= d_points_mapped[4]:
                """ Take fourth arc for calculations """
                arc_id = 3
            elif d_points_mapped[4] <= d_point <= d_points_mapped[5]:
                arc_id = 4
            elif d_points_mapped[5] <= d_point <= d_points_mapped[6]:
                arc_id = 5
            else:
                raise ValueError('Only 5 points considered in here!')

            z = self.get_gen_disc_point_height(arc_desc=route_v_desc[arc_id], d_point=d_point)
            generated_route_z_values.append({'d': d_point, 'z': z})

        if PLOT_FITNESS:
            plot_route_2d(plane=Plane.VERTICAL,
                          route_desc=route_v_desc,
                          route_len=self.fenotype.route_len_total,
                          p_dicts=generated_route_z_values)

        return [elem['z'] for elem in generated_route_z_values]

    def get_gen_disc_point_height(self, arc_desc: Dict[str, Any], d_point: float) -> np.int:
        """
        Calculates and returns height of certain point
        Params:                                                                     type:
        @param arc_desc: Description of arc                                         Dict[str, Any]
        @param d_point: Distance from start point                                   float
        @return: Height of certain point                                            np.int
        """

        d_center, z_center = arc_desc['point_cR'].coordinates
        direction = ArcDirection.ANTICLOCKWISE if arc_desc['arc_rad_len'] >= 0 else ArcDirection.CLOCKWISE

        alpha = np.arccos(float((d_point - d_center) / arc_desc['circle_radius_len']))
        if direction == ArcDirection.ANTICLOCKWISE:
            alpha += np.pi

        return np.int(arc_desc['circle_radius_len'] * np.sin(alpha) + z_center)

    def discretize_route(self) -> Union[List[Dict[str, Union[int, float]]], None]:
        """
        Returns points of discretized route
        Params:                                                                     type:
        @return: Points of discretized route                                        Union[List, None]
        """

        curr_distance = 0.0
        discrete_route_points = []

        for arc_desc in self.fenotype.route_desc_h:
            curr_distance, points = self.discretize_arc(arc_desc=arc_desc, curr_dist=curr_distance)
            discrete_route_points += points

        discrete_route_points.append({'x': X_WAW, 'y': Y_WAW, 'd': floor(self.fenotype.route_len_h)})

        in_region = self.is_route_in_region(route_points=discrete_route_points)

        if in_region is True:
            return discrete_route_points
        else:
            return None

    def discretize_arc(self, arc_desc: Dict[str, Any], curr_dist: float) -> Tuple[float, List]:
        """
        Returns points of discretized arc
        Params:                                                                     type:
        @param arc_desc: Description of arc                                         Dict[str, Any]
        @param curr_dist: Current distance                                          float
        @return: Discretized arc                                                    Tuple[float, List]
        """

        arc_rad_len = float(arc_desc['arc_rad_len'])
        circle_radius_len = float(arc_desc['circle_radius_len'])
        direction = ArcDirection.ANTICLOCKWISE if arc_rad_len >= 0 else ArcDirection.CLOCKWISE
        x_center, y_center = arc_desc['point_cR'].coordinates

        points = []
        distance = curr_dist
        arc_dist = 0.0
        ang_1km_fraction = DIST1KM / float(2 * sp.pi * circle_radius_len)
        ang_1km_rad = float(2 * np.pi * ang_1km_fraction)
        base_angle = float(dir_to_rad(arc_desc['ray_RA'].direction))

        n_1km_steps = ceil(abs(arc_rad_len) / ang_1km_rad)

        for p_id in range(n_1km_steps):
            if direction == ArcDirection.ANTICLOCKWISE:
                p_angle = base_angle + p_id * ang_1km_rad
            else:
                p_angle = base_angle - p_id * ang_1km_rad

            y = circle_radius_len * np.sin(p_angle) + y_center
            x = circle_radius_len * np.cos(p_angle) + x_center

            points.append({'x': x, 'y': y, 'd': distance})

            if p_id != n_1km_steps - 1:
                dist_temp = float(ang_1km_fraction * 2 * np.pi * circle_radius_len)
                distance += dist_temp
                arc_dist += dist_temp
            else:
                ang_rest = abs(arc_rad_len) - (n_1km_steps - 1) * ang_1km_rad
                rest_dist = float((ang_rest / (2 * np.pi)) * 2 * np.pi * circle_radius_len)
                distance += rest_dist
                arc_dist += rest_dist

        return distance, points
