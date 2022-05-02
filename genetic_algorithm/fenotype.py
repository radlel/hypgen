# -*- coding: utf-8 -*-
"""

This module includes definition of class Fenotype and its subclasses.

"""


from sympy import Point2D, Ray2D
from typing import List, Dict, Any

from geometry.route_gen import get_route_description
from definitions.types import *
from definitions.config import *


class ArcDesc:
    """ Definition of ArcDesc - curve description """
    def __init__(self) -> None:
        self.point_cR = Point2D(np.inf, np.inf)
        self.circle_radius_len = np.inf
        self.ray_RA = Ray2D(Point2D(0, 0), angle=0)
        self.arc_rad_len = np.inf
        self.ray_tangent_B = Ray2D(Point2D(0, 0), angle=0)
        self.segment = np.inf


class RouteDesc:
    """ Definition of RouteDesc - route description """
    def __init__(self) -> None:
        self.arcs = np.array([ArcDesc() for _ in range(CHROMOSOME_SIZE - 1)])


class Fenotype:
    """ Definition of Fenotype - description of route details and characteristics """
    def __init__(self) -> None:
        self.route_horizontal = RouteDesc()
        self.route_vertical = RouteDesc()
        self.route_len_h = np.inf
        self.route_len_total = np.inf
        self.route_desc_h = np.inf
        self.route_desc_v = np.inf
        self.fitness_val = np.inf

    def init_horizontal(self, p_dicts: List[Dict[str, int]], init_tangent: float) -> None:
        """
        Initializes horizontal route
        Params:                                                                     type:
        @param p_dicts: Consecutive route points                                    List[Dict[str, int]]
        @param init_tangent: Initial tangent in first route point                   float
        @return: None
        """

        route_desc, route_len = get_route_description(plane=Plane.HORIZONTAL, p_dicts=p_dicts,
                                                      init_tangent=init_tangent)
        arcs = self.route_horizontal.arcs
        for (arc, arc_desc) in zip(arcs, route_desc):
            arc.point_cR = arc_desc['point_cR']
            arc.circle_radius_len = arc_desc['circle_radius_len']
            arc.ray_RA = arc_desc['ray_RA']
            arc.arc_rad_len = arc_desc['arc_rad_len']
            arc.ray_tangent_B = arc_desc['ray_tangent_B']
            arc.segment = arc_desc['segment']

        self.route_len_h = route_len
        self.route_desc_h = route_desc

    def init_vertical(self, p_dicts: List[Dict[str, int]], init_tangent: float) -> None:
        """
        Initializes vertical route
        Params:                                                                     type:
        @param p_dicts: Consecutive route points                                    List[Dict[str, int]]
        @param init_tangent: Initial tangent in first route point                   float
        @return: None
        """

        route_desc, route_len = get_route_description(plane=Plane.VERTICAL, p_dicts=p_dicts, init_tangent=init_tangent)
        arcs = self.route_vertical.arcs
        for (arc, arc_desc) in zip(arcs, route_desc):
            arc.point_cR = arc_desc['point_cR']
            arc.ray_RA = arc_desc['ray_RA']
            arc.arc_rad_len = arc_desc['arc_rad_len']
            arc.ray_tangent_B = arc_desc['ray_tangent_B']
            arc.segment = arc_desc['segment']

        self.route_len_total = route_len
        self.route_desc_v = route_desc

    def get_route_desc(self, plane: Plane) -> List[Dict[str, Any]]:
        """
        Returns horizontal or vertical route description
        Params:                                                                     type:
        @param plane: Specifies vertical or horizontal plane                        Plane
        @return: Route description                                                  List[Dict[str, Any]]
        """

        if plane == Plane.HORIZONTAL:
            return self.route_desc_h
        elif plane == Plane.VERTICAL:
            return self.route_desc_v
        else:
            raise ValueError('Invalid plane parameter value! {}'.format(plane))
