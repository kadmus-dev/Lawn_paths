from lawn_paths.map_builder.shapefile import build_shapefile
from lawn_paths.map_builder.visualize import visualize
from lawn_paths.pipeline.pipeline import run


def get_masks(path):
    run(path, path)


def get_shapefile(path):
    build_shapefile(path, output_filename="./paths.shp")
    visualize("./paths.shp", "./map.html")
