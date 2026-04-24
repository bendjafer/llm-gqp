from typing import NamedTuple
import networkx as nx

OLLAMA_MODEL = "llama3.1:8b"
MAX_NODES = 50
MAX_EDGES = 100

GRAPH_GENERATORS = {
    "Karate Club": nx.karate_club_graph,
    "Les Miserables": nx.les_miserables_graph,
    "Petersen": nx.petersen_graph,
    "Florentine Families": nx.florentine_families_graph,
}

class SizeConfig(NamedTuple):
    node_count:       int
    edge_probability: float
    nearest_neighbors: int
    attachment_edges: int

SIZE_CONFIG = {
    "small":  SizeConfig(10,  0.30, 4, 2),
    "medium": SizeConfig(50,  0.10, 6, 3),
    "large":  SizeConfig(100, 0.05, 8, 4),
}

class Problem(NamedTuple):
    question:          str            # undirected (or shared) phrasing
    directed_question: str | None     # directed-specific; None = same as question
    format_rule:       str            # expected answer format
    few_shot_k:        int = 0        # per-problem default; 0 = disabled

P_PROBLEMS: dict[str, Problem] = {
    "node_count":    Problem("How many nodes does this graph have?",                         None,                                                         "integer only"),
    "edge_count":    Problem("How many edges does this graph have?",                         None,                                                         "integer only"),
    "neighbors":     Problem("What are the neighbors of the first node?",                    "What are the out-neighbors (successors) of the first node?",  "Python list of node identifiers, e.g. [0, 2, 5]"),
    "degree":        Problem("What is the degree of the first node?",                       "What is the total degree (in+out) of the first node?",        "integer only"),
    "shortest_path": Problem("What is the shortest path from the first node to the last?",  None,                                                         "Python list of every node on the path, e.g. [0, 3, 7]"),
    "clustering":    Problem("What is the clustering coefficient of the first node?",       None,                                                         "float rounded to 4 decimal places"),
    "diameter":      Problem("What is the diameter of this graph?",                         None,                                                         "integer only"),
    "density":       Problem("What is the density of this graph?",                          None,                                                         "float rounded to 4 decimal places"),
    "is_connected":  Problem("Is this graph connected?",                                    "Is this graph weakly connected (ignoring edge direction)?",   "True or False only"),
    "cycle":         Problem("Does this graph contain a cycle?",                             "Does this graph contain a directed cycle?",                   "True or False only"),
}

# Backward-compat alias so the notebook keeps working.
P_PROBLEMS_PROMPTS = {k: v.question for k, v in P_PROBLEMS.items()}

# Difficulty weights 1–10 (linear), inspired by the NPHardEval benchmark.
PROBLEM_DIFFICULTY = {
    "node_count": 1,
    "edge_count": 2,
    "neighbors": 3,
    "degree": 4,
    "shortest_path": 5,
    "clustering": 6,
    "diameter": 7,
    "density": 8,
    "is_connected": 9,
    "cycle": 10,
}