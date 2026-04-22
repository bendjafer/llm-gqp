import difflib
import networkx as nx
from config import GRAPH_GENERATORS

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

def ground_truth(graphs):
    gt = {}
    for name, G in graphs.items():
        nodes = list(G.nodes())
        first_node = nodes[0]
        last_node  = nodes[-1]
        gt[name] = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "neighbors": sorted(G.neighbors(first_node), key=str),
            "degree": G.degree(first_node),
            "shortest_path": nx.shortest_path(G, first_node, last_node) if nx.has_path(G, first_node, last_node) else None,
            "clustering": round(nx.clustering(G, first_node), 4),
            "diameter": nx.diameter(G) if nx.is_connected(G) else None,
            "density": round(nx.density(G), 4),
            "is_connected": nx.is_connected(G),
            "cycle": len(nx.cycle_basis(G)) > 0,
        }
    return gt