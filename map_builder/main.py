from shapefile import build_shapefile
from visualize import visualize
import timeit


def main():
    start = timeit.default_timer()

    build_shapefile(r'../dataset', output_filename="../results/paths.shp")
    visualize("../results/paths.shp", "../results/map.html")

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == "__main__":
    main()
