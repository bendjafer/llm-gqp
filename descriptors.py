import io
import math
import random

import networkx as nx

from config import MAX_NODES, MAX_EDGES


def gdl_text(G):
    lines = ["text description:"]
    for n in list(G.nodes())[:MAX_NODES]:
        neighbors = list(G.neighbors(n))
        lines.append(
            f"Node {n} → {', '.join(map(str, neighbors)) if neighbors else 'isolated'}."
        )
    if G.number_of_nodes() > MAX_NODES:
        lines.append(f"[+{G.number_of_nodes() - MAX_NODES} nodes omitted]")
    return "\n".join(lines)


def gdl_adjacency_list(G):  # name param removed — never used by callers
    lines = ["adjacency list:"]
    for n in list(G.nodes())[:MAX_NODES]:
        neighbors = " ".join(map(str, sorted(G.neighbors(n), key=str)))
        lines.append(f"{n}: {neighbors}")
    if G.number_of_nodes() > MAX_NODES:
        lines.append(f"[+{G.number_of_nodes() - MAX_NODES} nodes omitted]")
    return "\n".join(lines)


def gdl_edge_list(G):  # name param removed — never used by callers
    arrow = "->" if G.is_directed() else "--"
    lines = ["edge list:"]
    edges = list(G.edges())[:MAX_EDGES]
    lines.extend(f"{u} {arrow} {v}" for u, v in edges)
    if G.number_of_edges() > MAX_EDGES:
        lines.append(f"[+{G.number_of_edges() - MAX_EDGES} edges omitted]")
    return "\n".join(lines)


def gdl_gml(G):
    return f"GML\n" + "\n".join(nx.generate_gml(G))


def gdl_graphml(G):
    allowed_types = (int, float, str, bool)

    H = G.__class__()

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
    return "graphml:\n"+buffer.getvalue().decode()

def gdl_random_walk(G, seed=42):

    random.seed(seed)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    nodes = list(G.nodes())

    if n == 0:
        return "Random walk description:\n[empty graph]"

    # ── Dynamic parameter computation (inspired by WalkLM) ──────────────
    # Walk length: paper uses termination prob α ≈ 0.05 → E[length] = 1/α = 20
    # We scale down for graph size and prompting context
    # For n nodes, log2(n)+2 gives: n=10→5, n=50→8, n=100→9, n=1000→12
    walk_length = max(3, min(int(math.log2(n + 1)) + 2, 10))

    # Number of walks: paper uses N~3e5 for fine-tuning; for prompting use
    # enough to cover all nodes once (diverse starting points), capped by MAX_NODES
    num_walks = min(n, MAX_NODES)

    # ── Starting node selection: spread across graph ────────────────────
    # If graph is disconnected, pick from each component
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    start_nodes = []
    per_component = max(1, num_walks // len(components))
    for comp in components:
        comp_nodes = list(comp)
        random.shuffle(comp_nodes)
        start_nodes.extend(comp_nodes[:per_component])
    start_nodes = start_nodes[:num_walks]

    # ── Walk generation ──────────────────────────────────────────────────
    arrow = "→" if G.is_directed() else "—"
    walks = []
    for start in start_nodes:
        walk = [str(start)]
        current = start
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(str(current))
        walks.append(walk)

    # ── Format output ─────────────────────────────────────────────────────
    lines = [
        "Random walk description",
        f"Parameters: {len(walks)} walks × length {walk_length} (n={n}, m={m})",
    ]
    for i, walk in enumerate(walks, 1):
        lines.append(f"Walk {i:2d}: {(' ' + arrow + ' ').join(walk)}")

    return "\n".join(lines)

# (imports are at the top of the file)


def gdl_l2sp_paths(G, b=8, L=4, k=3, short_len=3, seed=42):
    """
    Path-LLM style descriptor based on L2SP (Long-to-Short Shortest Paths).

    Parameters
    ----------
    G : networkx.Graph or DiGraph
        Input graph.
    b : int
        Number of sampled source nodes.
    L : int
        Minimum length for a long shortest path.
    k : int
        Number of long shortest paths sampled per source-target pair.
    short_len : int
        Maximum length of each short path chunk.
    seed : int
        Random seed.

    Returns
    -------
    str
        Textual descriptor similar to your other GDLs.
    """
    rng = random.Random(seed)

    if G.number_of_nodes() == 0:
        return "[empty graph]"

    nodes = list(G.nodes())
    sampled_sources = rng.sample(nodes, min(b, len(nodes)))
    all_short_paths = []

    for s in sampled_sources:
        lengths = nx.single_source_shortest_path_length(G, s)

        candidates = [t for t, d in lengths.items() if d >= L and t != s]
        if not candidates:
            continue

        t = rng.choice(candidates)

        try:
            long_paths = list(nx.all_shortest_paths(G, source=s, target=t))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if len(long_paths) > k:
            long_paths = rng.sample(long_paths, k)

        for path in long_paths:
            n = len(path)
            if n <= short_len:
                all_short_paths.append(path)
                continue

            step = max(1, short_len - 1)
            for i in range(0, n, step):
                chunk = path[i:i + short_len]
                if len(chunk) >= 2:
                    all_short_paths.append(chunk)
                if i + short_len >= n:
                    break

    if not all_short_paths:
        lines = "[no valid long-to-short shortest paths found; graph may be too small or too dense for current parameters]"
        return lines

    arrow = " -> " if G.is_directed() else " -- "
    lines = [
        "Long-to-short shortest paths (L2SP)",
        f"Parameters: {len(all_short_paths)} paths × length {short_len} (b={b}, L={L}, k={k})",
    ]

    for i, path in enumerate(all_short_paths, 1):
        lines.append(f"Path {i}: {arrow.join(map(str, path))}")

    return "\n".join(lines)


DESCRIPTORS = {
    "text": gdl_text,
    "adjacency_list": gdl_adjacency_list,
    "edge_list": gdl_edge_list,
    "gml": gdl_gml,
    "graphml": gdl_graphml,
    "random_walk": gdl_random_walk,
    "l2sp_paths": gdl_l2sp_paths
}

