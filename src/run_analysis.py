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


class NeighbourPressureModel(AbstractModel):
    """
    This model computes the innovation/preservation probabilities
    based on the # of neighbours with a feature turned on. This
    should give rise to real 'lumpy' linguistic areas.
    To simplify things, we assume that language-internal and
    neighbour-induced probabilities are additive: each successive
    neighbour with the feature turned on provides the current
    value of the increment, which is then halved.
    """

    def __init__(
            self,
            neighbour_graph: nx.Graph,
            basic_innovation_probability: float,
            innovation_increment: float,
            basic_preservation_probability: float,
            preservation_increment: float,
            initial_state='negative'):
        super(NeighbourPressureModel, self).__init__(
            neighbour_graph, initial_state)
        self.basic_innovation_probability = basic_innovation_probability
        self.innovation_increment = innovation_increment
        self.basic_preservation_probability = basic_preservation_probability
        self.preservation_increment = preservation_increment

    def step(self):
        """Make one time step."""
        tmp_state = {}
        for node, state in self._state.items():
            if state == 1:
                on_probability = self.basic_preservation_probability
                running_increment = self.preservation_increment
            else:
                on_probability = self.basic_innovation_probability
                running_increment = self.innovation_increment
            for neighbour in self.graph.neighbors(node):
                if self._state[neighbour] == 1:
                    on_probability += running_increment
                    running_increment /= 2
            tmp_state[node] = int(random() < on_probability)
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


def run_and_record(model, n_iter, verbose=False, coords_dict=None, 
        model_name='model'):
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
            if coords_dict is not None:
                plot_model_state(model, coords_dict, f'../img/{model_name}_{i:0>6}.png')
    return state_vector


if __name__ == '__main__':
    from statistics import mean

    # Load data
    G = nx.read_gexf('../data/neighbour_graph.gexf')
    n_nodes = len(G.nodes())
    print(len(G.nodes()), 'nodes')

    # Simulation parameters
    n_iter = 400

    # Models
    # p_inno2 = 0.075
    # p_pres2 = 0.75
    # state_vector2 = run_and_record(IndependentModel(
    #     G, p_inno2, p_pres2, 'negative'), n_iter, True, coords_dict, 'indep')

    # inno_p_alone = 0.05
    # pres_p_alone = 0.2
    # inno_p_comp = 0.1
    # pres_p_comp = 0.8
    # state_vector3 = run_and_record(
    #     NeighbourAwareModel(
    #         G,
    #         innovation_probability_alone=inno_p_alone,
    #         preservation_probability_alone=pres_p_alone,
    #         innovation_probability_company=inno_p_comp,
    #         preservation_probability_company=pres_p_comp
    #     ),
    #     n_iter, True, coords_dict, 'binary')

    basic_inno_p = 0.01
    basic_pres_p = 0.01
    inno_inc = 0.13
    pres_inc = 0.35
    state_vector1 = run_and_record(
        NeighbourPressureModel(
            G,
            basic_inno_p,
            inno_inc,
            basic_pres_p,
            pres_inc),
        n_iter, True)

    print(
        f'Mean for '
        f'({basic_inno_p=}, {basic_pres_p=}, {inno_inc=}, {pres_inc=}): '
        f'{mean(state_vector1)}')

    basic_inno_p = 0.01
    basic_pres_p = 0.01
    inno_inc = 0.13
    pres_inc = 0.4
    state_vector2 = run_and_record(
        NeighbourPressureModel(
            G,
            basic_inno_p,
            inno_inc,
            basic_pres_p,
            pres_inc),
        n_iter, True)

    print(
        f'Mean for '
        f'({basic_inno_p=}, {basic_pres_p=}, {inno_inc=}, {pres_inc=}): '
        f'{mean(state_vector2)}')

    basic_inno_p = 0.01
    basic_pres_p = 0.01
    inno_inc = 0.13
    pres_inc = 0.45
    state_vector3 = run_and_record(
        NeighbourPressureModel(
            G,
            basic_inno_p,
            inno_inc,
            basic_pres_p,
            pres_inc),
        n_iter, True, coords_dict)

    # Perform grid search on inno_inc and pres_inc
    print('Performing grid search...')
    import numpy as np
    inc_range = list(np.arange(0.0, 0.46, 0.05))
    max_idx = len(inc_range) - 1
    grid_search_result = np.zeros((len(inc_range), len(inc_range)))
    from tqdm import tqdm
    for i, inno_inc in tqdm(list(enumerate(inc_range))):
        for j, pres_inc in enumerate(inc_range):
            state_vector = run_and_record(NeighbourPressureModel(
                G,
                basic_inno_p,
                inno_inc,
                basic_pres_p,
                pres_inc), 
            n_iter, False)
            grid_search_result[max_idx - i,j] = np.mean(state_vector[-100:]) / n_nodes

    np.savetxt('grid_search_result_001_001_045_045.csv', grid_search_result, delimiter=',')

    # plt.figure(figsize=(16, 12))
    # plt.imshow(grid_search_result, extent=[0.0, 0.45, 0.0, 0.45])
    # plt.xticks(inc_range)
    # plt.yticks(inc_range)
    # plt.savefig('../img/grid_search.png')


    # print(
    #     f'Mean for '
    #     f'({basic_inno_p=}, {basic_pres_p=}, {inno_inc=}, {pres_inc=}): '
    #     f'{mean(state_vector3)}')

    # print(
        # f'Mean for independent model ({p_inno2}, {p_pres2}): '
        # f'{mean(state_vector2)}')
    # print(
        # f'Mean for neighbour-aware model '
        # f'({inno_p_alone}, {pres_p_alone}, {inno_p_comp}, {pres_p_comp}): '
        # f'{mean(state_vector3)}')
    # print(
    #     f'Mean for '
    #     f'({basic_inno_p=}, {basic_pres_p=}, {inno_inc=}, {pres_inc=}): '
    #     f'{mean(state_vector4)}')

    plt.figure(figsize=(16, 10))
    plt.plot(list(range(1, n_iter + 1)), state_vector1)
    plt.plot(list(range(1, n_iter + 1)), state_vector2)
    plt.plot(list(range(1, n_iter + 1)), state_vector3)
    # TODO: add annotation to the plot.
    plt.savefig('../img/model_comparison.png')

    # plot_model_state(simple_model, coords_dict,
    #                  f'independent_25_50_step{i:0>5}.png')
