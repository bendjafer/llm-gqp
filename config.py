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

P_PROBLEMS_PROMPTS = {
    "node_count": "How many nodes does this graph have?",
    "edge_count": "How many edges does this graph have?",
    "neighbors": "What are the neighbors of the first node?",
    "degree": "What is the degree of the first node?",
    "shortest_path": "What is the shortest path from the first node to the last node?",
    "clustering": "What is the clustering coefficient of the first node?",
    "diameter": "What is the diameter of this graph?",
    "density": "What is the density of this graph?",
    "is_connected": "Is this graph connected?",
    "cycle": "Does this graph contain a cycle?",
}

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