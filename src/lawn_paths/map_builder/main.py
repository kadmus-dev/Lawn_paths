from lawn_paths.map_builder.shapefile import build_shapefile
from lawn_paths.map_builder.visualize import visualize


def main():
    build_shapefile(r'../../../dataset', output_filename="../../../results/paths.shp")
    visualize("../../../results/paths.shp", "../../../results/map.html")


if __name__ == "__main__":
    main()
