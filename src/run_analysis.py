from random import random
from copy import copy

import networkx as nx

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from prepare_neighbour_graph import coords_dict


class AbstractModel:
    """
    A base class with helper methods used by
    subclasses.
    """

    def __init__(
            self,
            neighbour_graph: nx.Graph,
            initial_state='negative'):
        self.graph = neighbour_graph
        self._state = {node: 0 for node in self.graph.nodes()}
        self._init_state(initial_state)

    def _init_state(self, initial_state):
        if initial_state == 'positive':
            for node in self.graph.nodes():
                self._state[node] = 1
        elif initial_state == 'negative':
            return
        else:
            for node in initial_state:
                self._state[node] = 1

    def get_state(self):
        """Returns the state of all nodes in the graph."""
        return copy(self._state)


class IndependentModel(AbstractModel):
    """
    This model assumes independence of languages. It is
    parameterised by a probability of innovating a feature,
    a probability of preserving a feature, and the initial
    state. The initial state can be one of the following:
    -- 'negative' (default): the feature is absent everywhere
    -- 'positive': the feature is present everywhere
    -- an iterable of nodes with the feature turned on.
    Nodes from the iterable not found in the graph will be
    silently ignored; passing an empty iterable has the same
    effect as passing 'negative'.
    """

    def __init__(
            self,
            neighbour_graph: nx.Graph,
            innovation_probability: float,
            preservation_probability: float,
            initial_state='negative'):
        super(IndependentModel, self).__init__(neighbour_graph, initial_state)
        self.innovation_probability = innovation_probability
        self.preservation_probability = preservation_probability

    def step(self):
        """Make one time step."""
        tmp_state = {
            node: int(random() < self.preservation_probability) if state == 1
            else int(random() < self.innovation_probability)
            for node, state in self._state.items()
        }
        self._state = tmp_state


class NeighbourAwareModel(AbstractModel):
    """
    This model updates node values based on the
    values of neighbouring nodes. Essentially, this
    is a probabilistic Game of Life. Innovation and
    preservation probabilities are split based on
    whether the language has at least one neighbour
    with the feature turned on or off.
    """

    def __init__(
            self,
            neighbour_graph: nx.Graph,
            innovation_probability_alone: float,
            preservation_probability_alone: float,
            innovation_probability_company: float,
            preservation_probability_company: float,
            initial_state='negative'):
        super(NeighbourAwareModel, self).__init__(
            neighbour_graph, initial_state)
        self.innovation_probability_alone = innovation_probability_alone
        self.preservation_probability_alone = preservation_probability_alone
        self.innovation_probability_company = innovation_probability_company
        self.preservation_probability_company = preservation_probability_company

    def step(self):
        """Make one time step."""
        tmp_state = {}
        for node, state in self._state.items():
            node_has_on_neighbours = False
            for neighbour in self.graph.neighbors(node):
                if self._state[neighbour] == 1:
                    node_has_on_neighbours = True
                    break
            if state == 1:
                if node_has_on_neighbours:
                    prob = self.preservation_probability_company
                else:
                    prob = self.preservation_probability_alone
            else:
                if node_has_on_neighbours:
                    prob = self.innovation_probability_company
                else:
                    prob = self.innovation_probability_alone
            tmp_state[node] = int(random() < prob)
        self._state = tmp_state


def plot_points_binary(points1, points2, coords_dict, filename):
    # Create points
    lats1 = []
    lons1 = []
    lats2 = []
    lons2 = []
    for lang, coords_tuple in coords_dict.items():
        lat, lon = coords_tuple
        if lang in points1:
            lats1.append(lat)
            lons1.append(lon)
        elif lang in points2:
            lats2.append(lat)
            lons2.append(lon)

    plt.figure(figsize=(32, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    plt.plot(lons1, lats1, marker='o', color='red', markersize=2,
             transform=ccrs.PlateCarree(), linewidth=0)
    plt.plot(lons2, lats2, marker='o', color='blue', markersize=2,
             transform=ccrs.PlateCarree(), linewidth=0)
    plt.savefig(filename)


def plot_model_state(model, coords_dict, filename):
    nodes_on = []
    nodes_off = []
    state_dict = model.get_state()
    for node, state in state_dict.items():
        if state == 1:
            nodes_on.append(node)
        else:
            nodes_off.append(node)
    plot_points_binary(nodes_on, nodes_off, coords_dict, filename)


def run_and_record(model, n_iter, verbose=False):
    state_vector = []
    for i in range(1, n_iter + 1):
        model.step()
        model_state = model.get_state()
        on = off = 0
        for node, state in model_state.items():
            if state == 1:
                on += 1
            else:
                off += 1
        state_vector.append(on)
        if verbose and (i == 1 or i % 100 == 0):
            print(f'Step {i: >7}: {on} nodes on; {off} nodes off')
    return state_vector


if __name__ == '__main__':
    # Load data
    G = nx.read_gexf('../data/neighbour_graph.gexf')
    print(len(G.nodes()), 'nodes')

    # Simulation parameters
    n_iter = 400

    # Models
    # p_inno1 = 0.25
    # p_pres1 = 0.25
    # state_vector1 = run_and_record(IndependentModel(
    #     G, p_inno1, p_pres1, 'negative'), n_iter, True)
    p_inno2 = 0.075
    p_pres2 = 0.75
    state_vector2 = run_and_record(IndependentModel(
        G, p_inno2, p_pres2, 'negative'), n_iter, True)

    inno_p_alone = 0.05
    pres_p_alone = 0.2
    inno_p_comp = 0.1
    pres_p_comp = 0.8
    state_vector3 = run_and_record(
        NeighbourAwareModel(
            G,
            innovation_probability_alone=inno_p_alone,
            preservation_probability_alone=pres_p_alone,
            innovation_probability_company=inno_p_comp,
            preservation_probability_company=pres_p_comp
        ),
        n_iter,
        True)

    from statistics import mean
    print(
        f'Mean for independent model ({p_inno2}, {p_pres2}): '
        f'{mean(state_vector2)}')
    print(
        f'Mean for neighbour-aware model '
        f'({inno_p_alone}, {pres_p_alone}, {inno_p_comp}, {pres_p_comp}): '
        f'{mean(state_vector3)}')

    plt.figure(figsize=(16, 10))
    plt.plot(list(range(1, n_iter + 1)), state_vector2)
    plt.plot(list(range(1, n_iter + 1)), state_vector3)
    # TODO: add annotation to the plot.
    plt.savefig('../img/model_comparison.png')

    # plot_model_state(simple_model, coords_dict,
    #                  f'independent_25_50_step{i:0>5}.png')
