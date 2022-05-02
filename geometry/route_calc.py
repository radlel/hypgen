# -*- coding: utf-8 -*-
"""

This module includes functions responsible for calculations
and creating plots of generated routes.

"""


import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mplp
import matplotlib.collections as mplc
from typing import List, Dict, Union, Any

from definitions.types import *


""" Use Agg to reduce memory consumption """
matplotlib.use('Agg')


def dir_to_rad(direction_point: sp.Point2D) -> Union[float, sp.atan]:
    """
    Converts point indicating direction to angle in radians
    Params:                                                                     type:
    @param direction_point: Indicates Ray direction in relation to point (0,0)  Point
    @return: Angle in radians                                                   Union[float, sp.atan]
    """

    x, y = direction_point.coordinates

    if x > 0 and y > 0:
        return sp.atan(y / x)
    elif x < 0 < y:
        return sp.pi + sp.atan(y / x)
    elif x < 0 and y < 0:
        return sp.pi + sp.atan(y / x)
    elif x > 0 > y:
        return 2 * sp.pi + sp.atan(y / x)
    elif x > 0 and y == 0:
        return 0
    elif x == 0 and y > 0:
        return sp.pi / 2
    elif x < 0 and y == 0:
        return sp.pi
    elif x == 0 and y < 0:
        return 3 / 2 * sp.pi
    else:
        raise ValueError('Invalid point coordinates: ({},{})!'.format(x, y))


def dict_to_points(p_dicts: List[Dict[str, int]], plane: Plane) -> List[sp.Point2D]:
    """
    Converts dictionary point representation to Point2d objects
    Params:                                                                     type:
    @param p_dicts: Consecutive route points                                    List[Dict[str, int]]
    @param plane: Specifies vertical or horizontal plane                        Plane
    @return: List of Point2D objects                                            List[sp.Point2D]
    """

    p_out = []

    for p_dict in p_dicts:
        if plane == Plane.HORIZONTAL:
            p_out.append(sp.Point2D(p_dict['x'], p_dict['y']))
        else:
            p_out.append(sp.Point2D(p_dict['d'], p_dict['z']))
    return p_out


def plot_route_2d(plane: Plane, route_desc: List[Dict[str, Any]], route_len: float, p_dicts: List[Dict[str, int]],
                  landform=None, title=None, z_min=None, z_max=None) -> None:
    """
    Creates route plot in 2D
    Params:                                                                     type:
    @param plane: Specifies vertical or horizontal plane                        Plane
    @param route_desc: Route description                                        List[Dict[str, Any]]
    @param route_len: Route length                                              float
    @param p_dicts: Consecutive route points                                    List[Dict[str, int]]
    @param landform: Describes landform shape                                   Union[None, (List, List)]
    @param title: Plot title                                                    Union[None, str]
    @param z_min: Minimum value on z axis                                       Union[None, float]
    @param z_max: Maximum value on z axis                                       Union[None, float]
    @return: None
    """

    ax = plt.gca()

    """ Set map limits """
    if plane == Plane.HORIZONTAL:
        ax.set_xlim(MAP_LIMIT['x_min'], MAP_LIMIT['x_max'])
        ax.set_ylim(MAP_LIMIT['y_min'], MAP_LIMIT['y_max'])
    elif plane == Plane.VERTICAL:
        ax.set_xlim(- 0.1 * route_len, 1.1 * route_len)
        """ Set minimum limit on plot y axis as -1000m to 3000m for better visibility """
        ax.set_ylim(min(-900, z_min) - 100, max(2900, z_max) + 100)

    for desc in route_desc:
        if desc['segment'] is None:
            """ Add arc to plot """
            x_r, y_r = desc['point_cR'].coordinates
            theta1 = np.degrees(float(dir_to_rad(desc['ray_RA'].direction)))
            theta2 = theta1 + np.degrees(float(desc['arc_rad_len']))
            if desc['arc_rad_len'] < 0:
                theta1, theta2 = theta2, theta1

            ax.add_patch(mplp.Arc((x_r, y_r), 2 * float(desc['circle_radius_len']),
                                  float(2 * desc['circle_radius_len']), theta1=float(theta1),
                                  theta2=float(theta2), edgecolor='b', lw=1.5))

            """ Add rays to plot """
            p_r, p_a = desc['ray_RA'].points
            p_rx, p_ry = p_r.coordinates
            p_ax, p_ay = p_a.coordinates
            p_bx, p_by = desc['ray_tangent_B'].source.coordinates
            segment = mplc.LineCollection([[(p_rx, p_ry), (p_ax, p_ay)], [(p_rx, p_ry), (p_bx, p_by)]], linewidths=1)
            ax.add_collection(segment)

            """ Add arc center point to plot """
            ax.scatter(p_rx, p_ry, cmap='viridis', linewidth=1)
            ax.text(p_rx, p_ry, 'r' + str(route_desc.index(desc)) + str(route_desc.index(desc) + 1), color='black')

        else:
            """ Add segment to plot """
            p1, p2 = desc['segment'].points
            p1x, p1y = p1.coordinates
            p2x, p2y = p2.coordinates
            segment = mplc.LineCollection([[(p1x, p1y), (p2x, p2y)]], linewidths=1.5)
            ax.add_collection(segment)

    """ Add boundary points to plot """
    for point in p_dicts:
        x, y = (point['x'], point['y']) if (plane == Plane.HORIZONTAL) else (point['d'], point['z'])

        ax.scatter(x, y, cmap='viridis', linewidth=1)
        if point == p_dicts[0]:
            ax.text(x, y, START_POINT_NAME, color='black')
        elif point == p_dicts[-1]:
            ax.text(x, y, END_POINT_NAME, color='black')
        else:
            ax.text(x, y, str(p_dicts.index(point)), color='black')
            pass

    """ Add landform for vertical """
    if landform is not None and plane == Plane.VERTICAL:
        d, z = landform
        plt.plot(d, z, label='Landform')
        plt.legend(loc="upper right")

    if plane == Plane.HORIZONTAL:
        ax.set_xlabel('Coordinates x [m]\n')
        ax.set_ylabel('Coordinates y [m]\n')
        plt.title('Horizontal: ' + title)
    else:
        ax.set_xlabel('Route length [m]')
        ax.set_ylabel('AMSL [m]')
        plt.title('Vertical: ' + title)

    ax.grid()

    plt.savefig(OUTPUT_PATH + title + '.png')
    plt.close()
