# -*- coding: utf-8 -*-
"""

This file includes types definitions and configuration parameters for route
start and end points.

"""


from enum import Enum
import numpy as np


""" Coordinates for route start and end points - Krakow and Warsaw """
X_KRK = 244000
Y_KRK = 567000
Z_KRK = 208
X_WAW = 487000
Y_WAW = 638000
Z_WAW = 108

X_MIN_ALLOWED = 194000
X_MAX_ALLOWED = 536000
Y_MIN_ALLOWED = 467000
Y_MAX_ALLOWED = 809000

DRAW_X_MIN = X_KRK
DRAW_X_MAX = X_WAW
DRAW_Y_MIN = Y_KRK
DRAW_Y_MAX = Y_WAW


""" Definitions of distances """
DIST1KM = 1000
DIST50KM = 50000
DIST100KM = 100000


""" Definitions for terrain classification """
TERRAIN_TUNNEL_HIGH = -6
TERRAIN_EXCAVATION_LOW = -6
TERRAIN_EXCAVATION_HIGH = -1
TERRAIN_GROUND_LOW = -1
TERRAIN_GROUND_HIGH = 1
TERRAIN_EMBANKMENT_LOW = 1
TERRAIN_EMBANKMENT_HIGH = 6
TERRAIN_PYLON_LOW = 6


""" Route creation parameters - angles """
ANGLE_MIN_H = 1 / 12 * np.pi
ANGLE_MAX_H = 5 / 12 * np.pi
ANGLE_MIN_V = - 1 / 32 * np.pi
ANGLE_MAX_V = 1 / 6 * np.pi


""" Definitions of min and max route vertical level """
ROUTE_MIN_HEIGHT = 0
ROUTE_MAX_HEIGHT = 500


""" Definition of start and end point """
START_POINT = {
    'x': X_KRK,
    'y': Y_KRK,
    'z': Z_KRK
}

END_POINT = {
    'x': X_WAW,
    'y': Y_WAW,
    'z': Z_WAW
}

START_POINT_NAME = 'KRK'
END_POINT_NAME = 'WAW'


""" Definitions for plotting """
PLOT_FITNESS = False
PLOT_SAVE = True


""" Penalties for exceeding route limitations """
PENALTY_TIGHT_ARC = 100
PENALTY_INVALID_ROUTE = 1000000


""" Costs of different types of infrastructure """
COST_TUBE = 27 / 1000
COST_TUNNEL_BASE = 26 / 1000
COST_PYLON_PARAM = 0.094 / 1000
COST_EXC = 1 / 1000
COST_EMB = 1 / 1000

MAX_COST = 100000000000000


""" Definition of map borders """
MAP_AREA_SIZE = 343 * DIST1KM

MAP_LIMIT = {
    'x_min': X_KRK - DIST50KM,
    'x_max': X_KRK - DIST50KM + MAP_AREA_SIZE,
    'y_min': Y_KRK - DIST100KM,
    'y_max': Y_KRK - DIST100KM + MAP_AREA_SIZE,
    'z_min': -1000,
    'z_max': 2000,
    'd_min': None,
    'd_max': None
}


class Plane(Enum):
    """ Definition of Planes - horizontal or vertical """
    NONE = 0,
    HORIZONTAL = 1,
    VERTICAL = 2,
    """ ========== """
    PLANE_NUM = 3


class ArcDirection(Enum):
    """ Definition of arc direction - straight, clockwise or anticlockwise """
    STRAIGHT = 0,
    CLOCKWISE = 1,
    ANTICLOCKWISE = 2,
    """ ========== """
    ARC_DIRECTION_NUM = 3


""" Definition of single point on the map """
POINT_DEF = {
    Plane.HORIZONTAL: {'x': np.inf, 'y': np.inf},
    Plane.VERTICAL: {'z': np.inf, 'd': np.inf}
}


""" Paths """
GEO_DATA_ALL_PATH = r'resources\geo_data\all\\'
GEO_DATA_OUT_REGION_PATH = r'resources\geo_data\out_region\\'
OUTPUT_PATH = r'output\\'


""" Geological data filenames definitions """
DATA_FILENAME_X = 'x.csv'
DATA_FILENAME_Y = 'y.csv'
DATA_FILENAME_Z = 'z.csv'
