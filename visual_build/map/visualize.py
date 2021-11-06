import mplleaflet
import geopandas as gpd
from matplotlib import pyplot as plt


def visualize(filename: str, output_file: str) -> None:
    """
    Build an interactive map visualizing data in a shapefile.
    matplotlib==3.3.2 required
    """
    predicted = gpd.read_file(filename).to_crs('epsg:4326')
    fig, ax = plt.subplots()
    predicted.geometry.plot(ax=ax, color='r')
    html = mplleaflet.fig_to_html(fig=fig)
    with open(output_file, "w") as f:
        f.write(html)
