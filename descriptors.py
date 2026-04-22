import io
import networkx as nx

from config import MAX_NODES, MAX_EDGES


def gdl_text(G, name="graph"):
    lines = [f"Graph '{name}':"]
    for n in list(G.nodes())[:MAX_NODES]:
        neighbors = list(G.neighbors(n))
        lines.append(
            f"Node {n} → {', '.join(map(str, neighbors)) if neighbors else 'isolated'}."
        )
    if G.number_of_nodes() > MAX_NODES:
        lines.append(f"[+{G.number_of_nodes() - MAX_NODES} nodes omitted]")
    return "\n".join(lines)


def gdl_adjacency_list(G, name=None):
    lines = [f"Graph '{name}' adjacency list:"]
    for n in list(G.nodes())[:MAX_NODES]:
        neighbors = " ".join(map(str, sorted(G.neighbors(n), key=str)))
        lines.append(f"{n}: {neighbors}")
    if G.number_of_nodes() > MAX_NODES:
        lines.append(f"[+{G.number_of_nodes() - MAX_NODES} nodes omitted]")
    return "\n".join(lines)


def gdl_edge_list(G, name=None):
    arrow = "->" if G.is_directed() else "--"
    lines = [f"Graph '{name}' edge list:"]
    edges = list(G.edges())[:MAX_EDGES]
    lines.extend(f"{u} {arrow} {v}" for u, v in edges)
    if G.number_of_edges() > MAX_EDGES:
        lines.append(f"[+{G.number_of_edges() - MAX_EDGES} edges omitted]")
    return "\n".join(lines)


def gdl_gml(G, name=None):
    return f"# Graph '{name}'\n" + "\n".join(nx.generate_gml(G))


def gdl_graphml(G, name=None):
    allowed_types = (int, float, str, bool)

    H = G.__class__()
    H.graph["name"] = name

    H.add_nodes_from(
        (node, {k: v for k, v in data.items() if isinstance(v, allowed_types)})
        for node, data in G.nodes(data=True)
    )

    H.add_edges_from(
        (u, v, {k: val for k, val in data.items() if isinstance(val, allowed_types)})
        for u, v, data in G.edges(data=True)
    )

    buffer = io.BytesIO()
    nx.write_graphml(H, buffer)
    return buffer.getvalue().decode()


DESCRIPTORS = {
    "text": gdl_text,
    "adjacency_list": gdl_adjacency_list,
    "edge_list": gdl_edge_list,
    "gml": gdl_gml,
    "graphml": gdl_graphml,
}

