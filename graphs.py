import os
import random
import difflib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import GRAPH_GENERATORS, SIZE_CONFIG


from typing import NamedTuple


class _Model(NamedTuple):
    builder:           object   # Callable(cfg, seed, directed) -> nx.Graph
    supports_directed: bool


_MODELS: dict[str, _Model] = {
    "erdos_renyi":     _Model(
        lambda cfg, seed, d: nx.gnp_random_graph(
            cfg.node_count, cfg.edge_probability, seed=seed, directed=d),
        supports_directed=True,
    ),
    "watts_strogatz":  _Model(
        lambda cfg, seed, d: nx.connected_watts_strogatz_graph(
            cfg.node_count, cfg.nearest_neighbors, p=0.2, seed=seed),
        supports_directed=False,
    ),
    "barabasi_albert": _Model(
        lambda cfg, seed, d: nx.barabasi_albert_graph(
            cfg.node_count, cfg.attachment_edges, seed=seed),
        supports_directed=False,
    ),
}


def load_graphs(graph_names=None):
    if graph_names is None:
        return {name: gen() for name, gen in GRAPH_GENERATORS.items()}
    graphs = {}
    for name in graph_names:
        if name not in GRAPH_GENERATORS:
            suggestions = difflib.get_close_matches(name, GRAPH_GENERATORS.keys(), n=2, cutoff=0.4)
            hint = f" Did you mean: {suggestions}?" if suggestions else f" Available: {list(GRAPH_GENERATORS.keys())}"
            print(f"[Warning] '{name}' not found.{hint}")
        else:
            graphs[name] = GRAPH_GENERATORS[name]()
    return graphs


def generate_graphs(model=None, size="small", directed=False, weighted=False, seed=None):
    if size not in SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Valid: {sorted(SIZE_CONFIG)}")

    if model is not None:
        if model not in _MODELS:
            raise ValueError(f"Unknown model '{model}'. Valid: {sorted(_MODELS)}")
        if directed and not _MODELS[model].supports_directed:
            raise ValueError(
                f"Model '{model}' is undirected-only. "
                f"Use 'erdos_renyi' for directed graphs, or omit --gen-model to auto-select."
            )
        spec = _MODELS[model]
    else:
        # Auto-select: first model that satisfies the directed constraint.
        candidates = [(n, s) for n, s in _MODELS.items() if not directed or s.supports_directed]
        if not candidates:
            raise ValueError(f"No registered model supports directed={directed}.")
        model, spec = candidates[0]

    cfg  = SIZE_CONFIG[size]
    rng  = random.Random(seed)
    name = (
        f"{model}"
        f"_{'directed' if directed else 'undirected'}"
        f"_{'weighted' if weighted else 'unweighted'}"
        f"_{size}_seed{seed}"
    )

    G = spec.builder(cfg, rng.randint(0, 10**9), directed)
    if weighted:
        nx.set_edge_attributes(G, {e: rng.randint(1, 10) for e in G.edges()}, "weight")

    return {name: G}


def save_graph_plots(graphs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, G in graphs.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        pos         = nx.spring_layout(G, seed=0)
        node_degree = [d for _, d in G.degree()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="#888888")
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_degree, cmap=plt.cm.plasma,
            node_size=300, alpha=0.9,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")

        plt.colorbar(nodes, ax=ax, label="Degree")

        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(
            f"Nodes: {G.number_of_nodes()}  |  "
            f"Edges: {G.number_of_edges()}  |  "
            f"Density: {nx.density(G):.3f}  |  "
            f"Connected: {nx.is_connected(G) if not G.is_directed() else 'N/A'}",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"graph_{name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Graph plot saved → {save_path}")


def ground_truth(graphs):
    gt = {}
    for name, G in graphs.items():
        nodes      = list(G.nodes())
        first_node = nodes[0]
        last_node  = nodes[-1]
        directed   = G.is_directed()

        is_conn  = nx.is_weakly_connected(G)   if directed else nx.is_connected(G)
        has_diam = nx.is_strongly_connected(G) if directed else is_conn

        gt[name] = {
            "node_count":    G.number_of_nodes(),
            "edge_count":    G.number_of_edges(),
            "neighbors":     sorted(list(G.successors(first_node)) if directed else list(G.neighbors(first_node)), key=str),
            "degree":        G.degree(first_node),
            "shortest_path": nx.shortest_path(G, first_node, last_node) if nx.has_path(G, first_node, last_node) else None,
            "clustering":    round(nx.clustering(G, first_node), 4),
            "diameter":      nx.diameter(G) if has_diam else None,
            "density":       round(nx.density(G), 4),
            "is_connected":  is_conn,
            "cycle":         any(True for _ in nx.simple_cycles(G)) if directed else len(nx.cycle_basis(G)) > 0,
        }
    return gt