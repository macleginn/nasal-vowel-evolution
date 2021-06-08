from math import radians

import pandas as pd
import numpy as np
import networkx as nx

from vincenty import vincenty
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import haversine_distances

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm import tqdm


def not_na(x):
    return not pd.isna(x)


def rev_t(tupl):
    return tupl[1], tupl[0]


# http://earthpy.org/tag/scipy.html
def lat_lon_to_cartesian(lat, lon, R=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return (x, y, z)


def get_haversine_distance(point1, point2):
    """A failsafe for when vincenty doesn't converge."""
    point1 = [radians(el) for el in point1]
    point2 = [radians(el) for el in point2]
    result = haversine_distances([point1, point2]) * 6371000 / 1000
    # The function returns a matrix of pairwise distances.
    return result[0][1]


def plot_graph(G_loc, coords_dict, filename):
    # Create a line collection from the graph
    lines = [[] for i in range(len(G_loc.edges()))]
    for i, edge in enumerate(G_loc.edges()):
        t, h = edge
        lines[i] = [
            rev_t(coords_dict[t]),
            rev_t(coords_dict[h])
        ]
    lc = LineCollection(lines, colors='brown', linewidths=0.5)

    # Create points
    lats = []
    lons = []
    for lang, coords_tuple in coords_dict.items():
        if lang in G_loc.nodes():
            lat, lon = coords_tuple
            lats.append(lat)
            lons.append(lon)

    plt.figure(figsize=(32, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    plt.plot(lons, lats, marker='o', color='red', markersize=2,
             transform=ccrs.PlateCarree(), linewidth=0)
    ax.add_collection(lc)
    plt.savefig(filename)


# Prepare exported data

metadata = pd.read_csv('../data/languages_and_dialects_geo.csv')

coords_dict = {}
for row in metadata.itertuples():
    if row.level != 'language' or\
            pd.isna(row.latitude) or\
            pd.isna(row.longitude):
        continue
    coords_dict[row.glottocode] = (row.latitude, row.longitude)


if __name__ == '__main__':
    phoible = pd.read_csv('../data/phoible.csv', low_memory=False)

    # Filter out languages without coordinates
    phoible = phoible.loc[phoible['Glottocode'].map(
        lambda x: not pd.isna(x))]
    phoible_w_coords = phoible.merge(
        metadata[['glottocode', 'latitude', 'longitude']],
        left_on='Glottocode', right_on='glottocode'
    )[[
        'InventoryID', 'Glottocode', 'LanguageName',
        'Phoneme', 'latitude', 'longitude'
    ]]
    phoible_w_coords = phoible_w_coords.loc[
        (
            phoible_w_coords['latitude'].map(not_na)
        ) & (phoible_w_coords['longitude'].map(not_na))]
    languages = set(phoible_w_coords.Glottocode.unique())

    # Compute the Delaunay triangulation

    cartesian_coords_dict = {
        k: lat_lon_to_cartesian(*v) for k, v in coords_dict.items()
    }
    name_arr = [el for el in sorted(coords_dict) if el in languages]
    name_dict = {name: i for i, name in enumerate(name_arr)}
    points_arr = [cartesian_coords_dict[lang] for lang in name_arr]
    tri = Delaunay(points_arr)
    indptr, indices = tri.vertex_neighbor_vertices

    G = nx.Graph()

    for k in tqdm(range(len(indptr) - 1)):
        point_gltc = name_arr[k]
        G.add_node(point_gltc)
        neighbours = indices[indptr[k]:indptr[k + 1]]
        for n in neighbours:
            neigh_gltc = name_arr[n]
            try:
                if vincenty(
                    coords_dict[point_gltc],
                    coords_dict[neigh_gltc]
                ) <= 500:
                    G.add_edge(point_gltc, neigh_gltc)
            except TypeError:
                # vincenty didn't converge
                if get_haversine_distance(
                    coords_dict[point_gltc],
                    coords_dict[neigh_gltc]
                ) <= 500:
                    G.add_edge(point_gltc, neigh_gltc)
    plot_graph(G, coords_dict, '../img/basic_neighbour_graph.png')

    # Dump the graph
    nx.write_gexf(G, '../data/neighbour_graph.gexf')
