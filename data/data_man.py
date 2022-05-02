# -*- coding: utf-8 -*-
"""

This module is responsible for geological data management.

"""


from typing import List
import pandas as pd

from definitions.types import *


def get_geo_data_element_z(z_data: np.array, x_coordinate: int, y_coordinate: int, x_range: List, y_range: List) -> int:
    """
    Returns z value for requested coordinates in geo data
    Params:                                                                     type:
    @param z_data: Axis z mesh                                                  np.array
    @param x_coordinate: Coordinate x                                           int
    @param y_coordinate: Coordinate y                                           int
    @param x_range: Range of axis x                                             List
    @param y_range: Range of axis y                                             List
    @return: z value                                                            int
    """

    return z_data[list(x_range).index(round(x_coordinate))][list(y_range).index(round(y_coordinate))]


def get_geo_data_all() -> (np.array, np.array, np.array):
    """
    Returns all available geo data
    Params:                                                                     type:
    @return: Geo data as x, y coordinates and z mesh                            (np.array, np.array, np.array)
    """

    x_data = np.array(pd.read_csv(GEO_DATA_ALL_PATH + DATA_FILENAME_X))
    y_data = np.array(pd.read_csv(GEO_DATA_ALL_PATH + DATA_FILENAME_Y))
    z_data = np.array(pd.read_csv(GEO_DATA_ALL_PATH + DATA_FILENAME_Z))

    x_data = np.array([elem[1] for elem in x_data])
    y_data = np.array([elem[1] for elem in y_data])
    z_data = np.array([elem[1:] for elem in z_data])

    return x_data, y_data, z_data


def get_geo_data_out_region_coordinates() -> (np.array, np.array):
    """
    Returns coordinates of mesh points out of considered region
    Params:                                                                     type:
    @return: Out of region geo data coordinates                                 (np.array, np.array)
    """

    x_inv = np.array(pd.read_csv(GEO_DATA_OUT_REGION_PATH + DATA_FILENAME_X))
    y_inv = np.array(pd.read_csv(GEO_DATA_OUT_REGION_PATH + DATA_FILENAME_Y))

    x_inv = np.array([elem[1] for elem in x_inv])
    y_inv = np.array([elem[1] for elem in y_inv])

    return x_inv, y_inv


class GeoData:
    """ Definition of GeoData - container for geological data """
    def __init__(self) -> None:
        self.__x, self.__y, self.__z = get_geo_data_all()
        self.__x_out, self.__y_out = get_geo_data_out_region_coordinates()

    def get_geo_data_all(self) -> (np.array, np.array, np.array):
        """
        Initializes horizontal route
        Params:                                                                     type:
        @return: Geo data as x, y coordinates and z mesh                            (np.array, np.array, np.array)
        """

        return self.__x, self.__y, self.__z

    def get_geo_data_out(self) -> (np.array, np.array):
        """
        Returns coordinates of mesh points out of considered region
        Params:                                                                     type:
        @return: Out of region geo data coordinates                                 (np.array, np.array)
        """

        return self.__x_out, self.__y_out
