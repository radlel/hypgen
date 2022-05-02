# -*- coding: utf-8 -*-
"""

This module includes functions responsible for geometric transformations
and generating routes.

"""


import sympy as sp
from typing import List, Dict, Any

from geometry.route_calc import dir_to_rad, dict_to_points
from definitions.types import *


def generate_route_desc_2d(points: List[sp.Point2D], init_tangent: float) -> List[Dict[str, Any]]:
    """
    Generates description for route arcs/segments
    Params:                                                                     type:
    @param points: Consecutive route points                                     List[sp.Point2D]
    @param init_tangent: Initial tangent in first route point                   float
    @return: List of route arcs/segments description dicts                      List[Dict[str, Any]]
    """

    desc = []
    ray_tangent = sp.Ray2D(points[0], angle=init_tangent)

    for p_id in range(len(points) - 1):
        point_a = points[p_id]
        point_b = points[p_id + 1]
        arc_desc = generate_arc_2d(point_a=point_a, point_b=point_b, ray_tangent_a=ray_tangent)
        ray_tangent = arc_desc['ray_tangent_B']
        desc.append(arc_desc)

    return desc


def generate_arc_2d(point_a: sp.Point2D, point_b: sp.Point2D, ray_tangent_a: sp.Ray2D) -> Dict[str, Any]:
    """
        Generates parameters needed for drawing an arc and tangent for next arc starting point
        Params:                                                                     type:
        @param point_a: Start point                                                 Point2D
        @param point_b: End point                                                   Point2D
        @param ray_tangent_a: Tangent in point A                                    Ray2D
        @return: Dictionary containing:                                             Dict[str, Any]
                    point_cR: Circle center,                                        Union[Point2D, None]
                    circle_radius_len: Radius length,                               Union[float, None]
                    ray_RA: Ray from circle center raising to point A,              Union[Ray2D, None]
                    arc_rad_len: Angle length of the arc in radians,                Union[float, None]
                    ray_tangent_B: Tangent in point B,                              Ray2D]
                    segment: Segment AB,                                            Union[Segment2D, None]
    """

    """ Check if AB is curve arc or segment """
    segment_ab = sp.Segment2D(point_a, point_b)
    if ray_tangent_a.is_parallel(segment_ab):
        """ AB is segment """

        """ Check if point B lies on rising direction of arc_tangent_A - if not then connection AB is impossible """
        if not ray_tangent_a.contains(point_b):
            raise ValueError('Forbidden combination! Point B lies in straight line behind point A. '
                             'Impossible to connect points! Point A: {}, Point B: {}'.format(point_a, point_b))

        """ Evaluate tangent in point B """
        ray_tangent_b = sp.Ray2D(point_b, angle=dir_to_rad(ray_tangent_a.direction))

        return {'point_cR': sp.Point2D(np.inf, np.inf),
                'circle_radius_len': np.inf,
                'ray_RA': np.inf,
                'arc_rad_len': np.inf,
                'ray_tangent_B': ray_tangent_b,
                'segment': segment_ab}

    else:
        """ AB is arc """

        """ Evaluate circle center """
        point_c_ab = segment_ab.midpoint
        line_c_abr = segment_ab.perpendicular_line(point_c_ab)
        line_ar = ray_tangent_a.perpendicular_line(point_a)
        point_c_r = line_ar.intersection(line_c_abr)[0]

        """ Evaluate length of circle radius """
        circle_radius_len = point_a.distance(point_c_r)

        """ Evaluate ray starting in circle center and rising in point A direction """
        ray_ra = sp.Ray2D(point_c_r, point_a)

        """ Evaluate angle length of the arc """
        ray_tang_a_rot_p90 = ray_tangent_a.rotate(angle=sp.pi / 2)
        ray_tang_a_rot_m90 = ray_tangent_a.rotate(angle=-sp.pi / 2)
        ray_ab = sp.Ray2D(point_a, point_b)

        tang_a_m90_limit = dir_to_rad(direction_point=ray_tang_a_rot_m90.direction)
        tang_a_p90_limit = dir_to_rad(direction_point=ray_tang_a_rot_p90.direction)
        dir_angle_ab = dir_to_rad(direction_point=ray_ab.direction)

        """ Check if point_cAB and point_cR are the same point - if yes then angle_ARB is pi """
        if point_c_ab.equals(point_c_r):
            angle_arb = sp.pi
        else:
            angle_arb = (sp.Triangle(point_a, point_c_r, point_b)).angles[point_c_r]

        if tang_a_m90_limit < tang_a_p90_limit:
            if tang_a_m90_limit < dir_angle_ab < tang_a_p90_limit:
                """ Short angle """
                arc_rad_len = angle_arb
            else:
                """ Long angle """
                arc_rad_len = 2 * sp.pi - angle_arb
        else:
            if dir_angle_ab < tang_a_p90_limit or dir_angle_ab > tang_a_m90_limit:
                """ Short angle """
                arc_rad_len = angle_arb
            else:
                """ Long angle """
                arc_rad_len = 2 * sp.pi - angle_arb

        """ Check rotation direction """
        if (((ray_ab.closing_angle(ray_tangent_a) < 0) and (abs(ray_ab.closing_angle(ray_tangent_a)) < sp.pi)) or
                (((ray_ab.closing_angle(ray_tangent_a)) > 0) and (abs(ray_ab.closing_angle(ray_tangent_a)) > sp.pi))):
            """ Clockwise direction - negative angle value """
            arc_rad_len = -arc_rad_len
        else:
            """ Counter clockwise direction - do nothing """
            pass

        """ Evaluate tangent in point B """
        dir_angle_tangent_a = dir_to_rad(direction_point=ray_tangent_a.direction)
        dir_angle_tangent_b = dir_angle_tangent_a + arc_rad_len
        ray_tangent_b = sp.Ray2D(point_b, angle=dir_angle_tangent_b)

        return {'point_cR': point_c_r,
                'circle_radius_len': circle_radius_len,
                'ray_RA': ray_ra,
                'arc_rad_len': arc_rad_len,
                'ray_tangent_B': ray_tangent_b,
                'segment': None}


def get_route_len_2d(route_desc: List[Dict[str, Any]]) -> float:
    """
    Calculates route length defined by parameter points and tangent in first point
    Params:                                                                     type:
    @param route_desc: Route description                                        List[Dict[str, Any]]
    @return: Route length                                                       float
    """

    route_len = 0
    for desc in route_desc:
        if desc['segment'] is not None:
            route_len += float(desc['segment'].length)
        else:
            route_len += float(2 * sp.pi * float(desc['circle_radius_len']) *
                               (abs(float(desc['arc_rad_len'] / (2 * sp.pi)))))

    return float(route_len)


def get_route_description(plane: Plane, p_dicts: List[Dict[str, int]], init_tangent: float) -> (List[Dict[str, Any]],
                                                                                                float):
    """
    Returns route description
    Params:                                                                     type:
    @plane: Specifies vertical or horizontal plane                              Plane
    @p_dicts: Consecutive route points                                          List[Dict[str, int]]
    @init_tangent: Initial tangent in first route point                         float
    @return: Route description and route length                                 (List[Dict[str, Any]], float)
    """

    points = dict_to_points(plane=plane, p_dicts=p_dicts)
    route_desc = generate_route_desc_2d(points=points, init_tangent=float(init_tangent))
    route_len = get_route_len_2d(route_desc=route_desc)

    return route_desc, route_len
