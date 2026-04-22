import argparse
import os
import sys

from config import OLLAMA_MODEL
from corpus import Corpus
from graphs import load_graphs
from llms_langchain import OllamaLLM

try:
    import ollama
except ImportError:
    ollama = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build/load graph descriptors, run Ollama on graph tasks, and save enriched results CSV."
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        required=True,
        help="List of graph names to load, e.g. --graphs Petersen 'Karate Club'",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        required=True,
        help="List of descriptor formats, e.g. --descriptors text adjacency_list edge_list",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model name. Defaults to config OLLAMA_MODEL ({OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--corpus-path",
        default="results/graphs_translation.json",
        help="Path to descriptor cache JSON.",
    )
    return parser.parse_args()


def model_exists(model_name: str) -> bool:
    if ollama is None:
        return True
    try:
        models_info = ollama.list()
        # ollama-py < 0.2 returns a dict; >= 0.2 returns a ListResponse object.
        models = models_info.get("models", []) if isinstance(models_info, dict) else getattr(models_info, "models", [])
        available = set()
        for m in models:
            if isinstance(m, dict):
                if "model" in m:
                    available.add(m["model"])
                if "name" in m:
                    available.add(m["name"])
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name:
                    available.add(name)
        return model_name in available
    except Exception:
        return True


def ensure_graphs_loaded(graph_names):
    graphs = load_graphs(graph_names)
    if not graphs:
        raise ValueError("No valid graphs were loaded. Check graph names.")
    missing = [g for g in graph_names if g not in graphs]
    if missing:
        print(f"[WARN] These graphs were not loaded: {missing}")
    return graphs


def ensure_corpus(graphs, descriptor_names, corpus_path):
    corpus = Corpus(path=corpus_path)

    corpus_exists = os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0
    if corpus_exists:
        corpus.load()
        all_present = all(
            name in corpus and all(fmt in corpus.get(name) for fmt in descriptor_names)
            for name in graphs
        )
        if all_present:
            print(f"[INFO] All requested descriptors already exist in {corpus_path}. Loading directly.")
            return corpus
        print(f"[INFO] Missing descriptors detected. Building only missing entries into {corpus_path}.")
        return corpus.build(graphs, descriptor_names)

    print(f"[INFO] Descriptor cache not found. Building descriptors into {corpus_path}.")
    return corpus.build(graphs, descriptor_names)


def main():
    args = parse_args()
    model_name = args.model or OLLAMA_MODEL

    if not model_exists(model_name):
        print(f"[ERROR] Ollama model '{model_name}' does not exist locally.")
        print("Run 'ollama list' to inspect installed models or pull the requested model first.")
        sys.exit(1)

    try:
        graphs = ensure_graphs_loaded(args.graphs)
        corpus = ensure_corpus(graphs, args.descriptors, args.corpus_path)
        llm = OllamaLLM(model=model_name)

        llm.generate_answers(
            corpus,
            graphs,
            selected_graphs=args.graphs,
            selected_descriptors=args.descriptors,
        )
        
        print(f"[DONE] Results available at: {llm.save_path}")
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
