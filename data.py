import numpy as np
import rasterio as rio
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

WINDOW = (slice(100, 300), slice(400, 600))

YEARS = [1990, 2000, 2006, 2012, 2018]

# [latifoglie cespugli aree_nude erba conifere coltivi faggete]
G_INDEX = 3  # Grassland
S_INDEX = 1  # Shrubland
C_INDEX = 4  # Coniferous
B_INDEX = 0  # Broadleaved
O_INDEX = 2  # Other

def load_data():
    data = {}
    for y in YEARS:
        with rio.open(f"data/clc_{y}.tif") as src:
            raster = src.read(1)
            raster = raster[WINDOW]
            data[y] = grid_from_raster(raster)
    return data


def create_colormap():
    colors = {
        G_INDEX: "green",
        S_INDEX: "yellow",
        C_INDEX: "yellowgreen",
        B_INDEX: (0.5, 0.25, 0),
        2: "gray",
        5: "green",
        6: (0.5, 0.25, 0),
        7: "black",
    }
    cmap = mcolors.ListedColormap([colors[i] for i in range(len(colors))])
    bounds = [i - 0.5 for i in range(len(colors) + 1)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


cmap, norm = create_colormap()


def plot_data():
    for y in YEARS:
        with rio.open(f"data/clc_{y}.tif") as src:
            raster = src.read(1)
            plt.figure()
            plt.imshow(raster[WINDOW], cmap=cmap, norm=norm)

            plt.title(f"Land cover {y}")
            plt.show()


#plot_data()
