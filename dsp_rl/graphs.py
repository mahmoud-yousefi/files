"""Graph generators and the graph container used by the environment and the model."""
from dataclasses import dataclass
import numpy as np
import networkx as nx


@dataclass
class GraphData:
    n: int
    row: np.ndarray      # directed edge list (both directions), source index
    col: np.ndarray      # directed edge list, target index; message flows col -> row
    an: np.ndarray       # symmetric-normalised adjacency value per directed edge (DECISIONS D5)
    deg: np.ndarray      # float32 degree
    nbrs: list           # list of np.ndarray, open neighbourhoods


def make_graph_data(G: nx.Graph) -> GraphData:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    n = G.number_of_nodes()
    e = np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2)
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(row, minlength=n).astype(np.float32)
    an = (1.0 / np.sqrt(deg[row] * deg[col])).astype(np.float32) if len(row) else np.zeros(0, np.float32)
    nbrs = [np.array(sorted(G[v]), dtype=np.int64) for v in range(n)]
    return GraphData(n, row, col, an, deg, nbrs)


def _seed(rng):
    return int(rng.integers(2**31 - 1))


def er(n, p, rng):  return nx.fast_gnp_random_graph(n, p, seed=_seed(rng))
def ba(n, m, rng):  return nx.barabasi_albert_graph(n, m, seed=_seed(rng))
def grid(m, n):     return nx.grid_2d_graph(m, n)
def tri(m, n):      return nx.triangular_lattice_graph(m, n)
def hexa(m, n):     return nx.hexagonal_lattice_graph(m, n)
# exp8 sparse families: every tree has >= 2 leaves, so the forced set F is never empty
def tree(n, rng):   return nx.random_labeled_tree(n, seed=_seed(rng))   # uniform random labeled tree
def ba1(n, rng):    return nx.barabasi_albert_graph(n, 1, seed=_seed(rng))  # random recursive tree (m=1)

GENERATORS = {"er": er, "tree": tree, "ba1": ba1}   # by name, for train.py --graph


def sample(a, n, rng):
    """A graph of family `a.graph` on n vertices (train.py --graph; default ER)."""
    if a.graph == "er":
        return er(n, a.p, rng)
    return GENERATORS[a.graph](n, rng)
