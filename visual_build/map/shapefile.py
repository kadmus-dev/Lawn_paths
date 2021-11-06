from math import fabs
from os import listdir
from os.path import isfile, join
from typing import List, Optional

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from rdp import rdp
from shapely.geometry import Polygon, LineString


class PixelPolygon(object):
    """
    Class for building polygon by its border pixels row by row.
    """

    def __init__(self, max_pixel_path_distance: int):
        """
        :param max_pixel_path_distance: max number of pixels between 2 pixels for them to be connected
        """
        self.__max_pixel_path_distance = max_pixel_path_distance
        self.__sequence = list()

    def __empty(self):
        return len(self.__sequence) == 0

    def __first(self):
        return None if self.__empty() else self.__sequence[0]

    def __last(self):
        return None if self.__empty() else self.__sequence[-1]

    @staticmethod
    def __is_close(point1, point2, max_pixel_path_distance):
        distance = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        return distance <= max_pixel_path_distance

    def add_segment(self, segment: ((int, int), (int, int))) -> bool:
        """
        Add a pixel row defined by ist border pixels.

        :param segment: tuple of 2 pixels (their coordinates)
        :return: True if any of the pixels was added to the polygon
        """
        if self.__empty() or self.__is_close(self.__first(), segment[0], self.__max_pixel_path_distance) or \
                self.__is_close(self.__last(), segment[1], self.__max_pixel_path_distance):
            self.__sequence.insert(0, segment[0])
            if segment[0] != segment[1]:
                self.__sequence.append(segment[1])
            return True
        return False

    def coord_transform(self, x_func, y_func) -> None:
        """
        Apply x_func to all x coordinates of pixels, y_func to all y coordinates of pixels
        """
        self.__sequence = [(x_func(x), y_func(y)) for y, x in self.__sequence]

    def build_centerline(self, p_epsilon: float = 0.3, c_epsilon: float = 0.3) -> Optional[LineString]:
        """
        Build centerline of the polygon.

        :param p_epsilon: RDP parameter to smooth polygon
        :param c_epsilon: RDP parameter to smooth centerline
        :return: shapely.geometry.Linestring if polygon was built correctly else None
        """
        try:
            polygon = Polygon(rdp(self.__sequence, p_epsilon))
            return LineString(rdp(sorted(polygon.exterior.coords, key=lambda x: x[1]), c_epsilon))
        except ValueError:
            return None


class PathImage(object):
    """
    Class for processing numpy image masks.
    """

    def __init__(self, filename: str,
                 max_path_distance_cm: float = 100,
                 min_bbox_size_m: float = 1,
                 max_bbox_size_m: float = 200):
        """
        :param filename: name of .NPY file with image mask
        :param max_path_distance_cm: max distance between paths for them to be connected in cm
        :param min_bbox_size_m: min size of path's bounding box in meters
        :param max_bbox_size_m: maxsize of path's bounding box in meters
        """
        self.__image = np.load(filename + '.npy')
        self.__min_bbox_size_m = min_bbox_size_m
        self.__max_bbox_size_m = max_bbox_size_m
        with open(filename + '.tfw', 'r') as f:
            data = f.read().split('\n')
            self.__x_transform = lambda x: float(data[4]) + x * float(data[0])
            self.__y_transform = lambda y: float(data[5]) + y * float(data[3])
            self.__max_pixel_path_distance = max_path_distance_cm // (fabs(float(data[0])) * 100)

    def is_empty(self) -> bool:
        """
        :return True if there are no path (non-zero) pixels in the mask
        """
        return not self.__image.any()

    def plot(self, figsize: (int, int) = (8, 8)) -> None:
        """
        Plot image mask with set figsize.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.__image, interpolation='nearest')
        plt.show()

    @staticmethod
    def __merge_paths(paths, max_pixel_path_distance):
        merged = list()
        for path in paths:
            if not merged or path[0] - merged[-1][1] > max_pixel_path_distance:
                merged.append(path)
            else:
                merged.append((merged[-1][0], path[1]))
                merged.pop(-2)
        return merged

    def __find_image_paths(self):
        image_paths = list()
        for i in range(self.__image.shape[0]):
            pixel_row = self.__image[i]
            if not pixel_row.any():
                continue
            path_pixels = np.nonzero(pixel_row)[0]
            row_gaps = [0] + list(np.where(path_pixels[1:] - path_pixels[:-1] - 1 != 0)[0] + 1) + [path_pixels.size]
            row_paths = [(path_pixels[row_gaps[j]], path_pixels[row_gaps[j + 1] - 1]) for j in range(len(row_gaps) - 1)]
            image_paths.append((i, self.__merge_paths(row_paths, self.__max_pixel_path_distance)))
        return image_paths

    def __check_bbox(self, centerline):
        if centerline is None:
            return None
        bbox = centerline.bounds
        if bbox[2] - bbox[0] < self.__min_bbox_size_m and bbox[3] - bbox[1] < self.__min_bbox_size_m:
            return None
        if bbox[2] - bbox[0] > self.__max_bbox_size_m or bbox[3] - bbox[1] > self.__max_bbox_size_m:
            return None
        return centerline

    def get_paths(self, p_epsilon: float = 0.3, c_epsilon: float = 0.3) -> List[Optional[LineString]]:
        """
        Get list of all paths on the image.

        :param p_epsilon: RDP parameter to smooth path polygons
        :param c_epsilon: RDP parameter to smooth their centerlines
        :return list of shapely.geometry.Linestring representing paths
        """
        polygons = list()
        for row_paths in self.__find_image_paths():
            lat = row_paths[0]
            for lon1, lon2 in row_paths[1]:
                segment = ((lat, lon1), (lat, lon2))
                for polygon in polygons:
                    if polygon.add_segment(segment):
                        break
                else:
                    new_polygon = PixelPolygon(self.__max_pixel_path_distance)
                    new_polygon.add_segment(segment)
                    polygons.append(new_polygon)

        for polygon in polygons:
            polygon.coord_transform(self.__x_transform, self.__y_transform)
        centerlines = [polygon.build_centerline(p_epsilon, c_epsilon) for polygon in polygons]
        centerlines = [self.__check_bbox(centerline) for centerline in centerlines]
        return [centerline for centerline in centerlines if centerline is not None]


def build_shapefile(dataset_directory, file_list=None,
                    output_filename: str = 'paths.shp',
                    crs: str = 'epsg:32637',
                    max_path_distance_cm: float = 100,
                    min_bbox_size_m: float = 1,
                    max_bbox_size_m: float = 200,
                    p_epsilon: float = 0.3,
                    c_epsilon: float = 0.3):
    """
    Build shapefile containing paths of all given images.

    :param dataset_directory: directory where .NPY mask files and .TFW world files are contained
    :param file_list: list of filenames to be processed (without extensions)
    :param output_filename: name of the output file (should be .SHP)
    :param crs: initial coordinate reference system
    :param max_path_distance_cm: max distance between paths for them to be connected in cm
    :param min_bbox_size_m: min size of path's bounding box in meters
    :param max_bbox_size_m: maxsize of path's bounding box in meters
    :param p_epsilon: RDP parameter to smooth path polygons
    :param c_epsilon: RDP parameter to smooth their centerlines
    """

    # get all files from dataset directory
    if file_list is None:
        file_list = [f for f in listdir(dataset_directory) if isfile(join(dataset_directory, f))]
        file_list = sum([f.split('.') for f in file_list], [])
        file_list = {f for f in file_list if f not in {"npy", "tfw", "tif", "prj"}}

    images = [PathImage(join(dataset_directory, filename),
                        max_path_distance_cm=max_path_distance_cm,
                        min_bbox_size_m=min_bbox_size_m,
                        max_bbox_size_m=max_bbox_size_m) for filename in file_list]
    images = [image for image in images if not image.is_empty()]
    centerlines = sum([image.get_paths(p_epsilon=p_epsilon, c_epsilon=c_epsilon) for image in images], [])
    geometry = gpd.GeoSeries(centerlines)
    geometry.crs = crs
    gpd.GeoDataFrame(geometry, columns=['geometry']).to_file(output_filename)
