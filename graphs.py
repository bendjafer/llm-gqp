import io
import json
import random
import networkx as nx
# ── Graphs ───────────────────────────────────────────────────────────

def load_graphs():
    return {
        "Karate Club":          nx.karate_club_graph(),
        "Les Miserables":       nx.les_miserables_graph(),
        "Petersen":             nx.petersen_graph(),
        "Florentine Families":  nx.florentine_families_graph(),
    }


# ── Corpus ───────────────────────────────────────────────────────────

def build_corpus(graphs):
    """Run all descriptors on all graphs. Returns corpus[graph_name][format] = str."""
    corpus = {}
    for name, G in graphs.items():
        corpus[name] = {fmt: fn(G, name=name) for fmt, fn in DESCRIPTORS.items()}
    return corpus



def save_corpus(corpus, path="gdl_corpus.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Corpus saved → {path}  ({len(corpus)} graphs × {len(next(iter(corpus.values())))} formats)")


def ground_truth(graphs):
    gt = {}
    for name, G in graphs.items():
        nodes  = list(G.nodes())
        n0     = nodes[0]
        n1     = nodes[3] if len(nodes) > 3 else nodes[-1]

        gt[name] = {
            "node_count":    G.number_of_nodes(),
            "edge_count":    G.number_of_edges(),
            "neighbors":     sorted(G.neighbors(n0), key=str),
            "degree":        G.degree(n0),
            "shortest_path": nx.shortest_path(G, n0, n1),
            "clustering":    round(nx.clustering(G, n0), 4),
            "diameter":      nx.diameter(G),
            "density":       round(nx.density(G), 4),
            "is_connected":  nx.is_connected(G),
            "cycle":         len(nx.cycle_basis(G)) > 0,
        }
    return gt
# ── Descriptors ───────────────────────────────────────────────────────────

_MAX_NODES = 50


def gdl_text(G, name="graph"):
    lines = [f"Graph '{name}' has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."]
    for node in list(G.nodes())[:_MAX_NODES]:
        nb = list(G.neighbors(node))
        if not nb:
            lines.append(f"Node {node} has no connections.")
        elif len(nb) == 1:
            lines.append(f"Node {node} is connected to Node {nb[0]}.")
        else:
            lines.append(f"There are {len(nb)} nodes connected to Node {node}: {', '.join(str(v) for v in nb)}.")
    if G.number_of_nodes() > _MAX_NODES:
        lines.append(f"[Truncated: {G.number_of_nodes() - _MAX_NODES} more nodes not shown.]")
    return "\n".join(lines)


def gdl_adjacency_list(G, name=None):
    return "\n".join(
        f"N({node}) = {{{', '.join(str(v) for v in sorted(G.neighbors(node), key=str))}}}"
        for node in G.nodes()
    )


def gdl_edge_list(G, name=None):
    return "\n".join(f"({u}, {v})" for u, v in G.edges())


def gdl_cypher(G, name="G"):
    def sid(n):
        s = str(n).replace(" ", "_").replace("-", "_").replace(".", "_")
        return ("n" + s) if s and s[0].isdigit() else s

    arrow = "->" if G.is_directed() else "-"
    seen = set()
    rels = []
    for u, v in G.edges():
        key = frozenset({u, v})
        if key not in seen:
            seen.add(key)
            rels.append(f"(n{sid(u)})-[:CONNECTED_TO]-{arrow}(n{sid(v)})")

    nodes = [f"(n{sid(n)}:Node {{id: '{n}'}})" for n in G.nodes()]
    return "// Nodes\n" + "\n".join(nodes) + "\n\n// Relationships\n" + "\n".join(rels)


def gdl_gml(G, name=None):
    buf = io.StringIO()
    for line in nx.generate_gml(G):
        buf.write(line + "\n")
    return buf.getvalue()


def gdl_graphml(G, name=None):
    _ok = (int, float, str, bool)
    G2 = G.__class__()
    G2.add_nodes_from((n, {k: v for k, v in d.items() if isinstance(v, _ok)}) for n, d in G.nodes(data=True))
    G2.add_edges_from((u, v, {k: v for k, v in d.items() if isinstance(v, _ok)}) for u, v, d in G.edges(data=True))
    buf = io.BytesIO()
    nx.write_graphml(G2, buf)
    return buf.getvalue().decode()


def gdl_story(G, name="graph"):
    nodes, edges = list(G.nodes()), list(G.edges())
    hub, hub_deg = max(G.degree(), key=lambda x: x[1]) if nodes else (None, 0)
    first = f"{edges[0][0]} and {edges[0][1]} were the first to meet. " if edges else ""
    return (
        f"Once upon a time, in a network called '{name}', there were {len(nodes)} members. "
        f"{first}Over time, {len(edges)} connections formed. "
        f"The most connected member was node {hub}, who knew {hub_deg} others."
    )


def gdl_random_walk(G, name=None, num_walks=10, walk_length=10, seed=42):
    nodes = list(G.nodes())
    if not nodes:
        return ""
    rng = random.Random(seed)
    walks = []
    for _ in range(num_walks):
        cur = rng.choice(nodes)
        path = [cur]
        for _ in range(walk_length - 1):
            nb = list(G.neighbors(cur))
            if not nb:
                break
            cur = rng.choice(nb)
            path.append(cur)
        walks.append(" ".join(f"<node_{v}>" for v in path))
    return "\n".join(walks)


DESCRIPTORS = {
    "text":           gdl_text,
    "adjacency_list": gdl_adjacency_list,
    "edge_list":      gdl_edge_list,
    "cypher":         gdl_cypher,
    "gml":            gdl_gml,
    "graphml":        gdl_graphml,
    "story":          gdl_story,
    "random_walk":    gdl_random_walk,
}

