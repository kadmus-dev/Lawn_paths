from shapefile import build_shapefile
from fscore import f_score
from visualize import visualize


def main():
    build_shapefile('dataset', output_filename="output/paths.shp")
    print("F-score: %f" % f_score('output/standard.shp', 'output/neutral.shp', 'output/paths.shp'))
    # visualize("output/paths.shp", "map.html")


if __name__ == "__main__":
    main()
