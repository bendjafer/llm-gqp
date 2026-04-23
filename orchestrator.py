import argparse
import os
import sys

from config import OLLAMA_MODEL, SIZE_CONFIG
from descriptors import DESCRIPTORS
from corpus import Corpus
from graphs import load_graphs, generate_graphs, save_graph_plots
from llm import LLM, _detect_provider
from eval import save_eval_plots

try:
    import ollama
except ImportError:
    ollama = None


def _graph_dir(graph_names: list) -> str:
    label = graph_names[0] if len(graph_names) == 1 else "+".join(sorted(graph_names))
    return os.path.join("results", label)

def _run_dir(graph_names: list, model_name: str) -> str:
    safe = model_name.replace(":", "_").replace("/", "_")
    return os.path.join(_graph_dir(graph_names), safe)

def _corpus_path(graph_names: list) -> str:
    return os.path.join(_graph_dir(graph_names), "graph_translation.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build/load graph descriptors, run Ollama on graph tasks, and save enriched results CSV."
    )

    parser.add_argument("--graphs", nargs="*", default=[],
        help="Famous graph names, e.g. --graphs Petersen 'Karate Club'")

    parser.add_argument("--generate", action="store_true",
        help="Generate one random graph instead of loading a famous one.")
    parser.add_argument("--gen-model", default=None,
        choices=["erdos_renyi", "watts_strogatz", "barabasi_albert"],
        help="Random graph model. Auto-selected if omitted.")
    parser.add_argument("--gen-size", default="small",
        choices=sorted(SIZE_CONFIG),
        help="Size preset (default: small).")
    parser.add_argument("--gen-directed", action="store_true",
        help="Generate directed graphs (erdos_renyi only).")
    parser.add_argument("--gen-weighted", action="store_true",
        help="Assign random integer weights to edges.")
    parser.add_argument("--gen-seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42).")

    parser.add_argument("--descriptors", nargs="+", default=None,
        help="Descriptor formats (default: all available). "
             f"Available: {sorted(DESCRIPTORS)}")
    parser.add_argument("--model", default=None, help=(
        f"Model to use (default: {OLLAMA_MODEL}). "
        "Provider is auto-detected from the model name, or set LLM_PROVIDER in .env. "
        "Available models by provider:\n"
        "  OpenAI  (needs OPENAI_API_KEY) : gpt-4o  gpt-4o-mini  gpt-3.5-turbo  o1-mini  o3-mini\n"
        "  Gemini  (needs GEMINI_API_KEY) : gemini-2.0-flash  gemini-1.5-pro  gemini-1.5-flash\n"
        "  Groq    (needs GROQ_API_KEY, set LLM_PROVIDER=groq) : "
        "llama-3.3-70b-versatile  llama-3.1-8b-instant  mixtral-8x7b-32768  gemma2-9b-it\n"
        "  Ollama  (local, no key needed) : llama3.1:8b  gemma3:4b  phi4-mini  mistral:7b  qwen2.5:7b"
    ))
    parser.add_argument("--corpus-path", default=None,
        help="Override path to descriptor cache JSON. Auto-derived from graph names if omitted.")

    return parser.parse_args()


def model_exists(model_name: str) -> bool:
    if ollama is None:
        return True
    try:
        models_info = ollama.list()
        models    = models_info.get("models", []) if isinstance(models_info, dict) else getattr(models_info, "models", [])
        available = set()
        for m in models:
            if isinstance(m, dict):
                available.update(filter(None, [m.get("model"), m.get("name")]))
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

    if not corpus_exists:
        print(f"[INFO] Descriptor cache not found. Building descriptors into {corpus_path}.")
        return corpus.build(graphs, descriptor_names)

    # Load once to check completeness before deciding whether to rebuild.
    corpus.load()
    all_present = all(
        name in corpus and all(fmt in corpus.get(name) for fmt in descriptor_names)
        for name in graphs
    )
    if all_present:
        print(f"[INFO] All requested descriptors already exist in {corpus_path}. Loading directly.")
        return corpus

    print(f"[INFO] Missing descriptors detected. Building only missing entries into {corpus_path}.")
    return corpus.build(graphs, descriptor_names)  # build() re-reads — acceptable for a small JSON


def main():
    args       = parse_args()
    model_name = args.model or OLLAMA_MODEL

    graphs = {}

    if args.graphs:
        graphs.update(ensure_graphs_loaded(args.graphs))

    if args.generate:
        generated = generate_graphs(
            model    = args.gen_model,
            size     = args.gen_size,
            directed = args.gen_directed,
            weighted = args.gen_weighted,
            seed     = args.gen_seed,
        )
        graphs.update(generated)
        print(f"[INFO] Generated graph: {list(generated)[0]}")

    if not graphs:
        print("[ERROR] No graphs specified. Use --graphs and/or --generate.")
        sys.exit(1)

    graph_names      = list(graphs)
    descriptor_names = args.descriptors or sorted(DESCRIPTORS)
    if args.descriptors is None:
        print(f"[INFO] --descriptors not specified. Using all: {descriptor_names}")

    graph_dir   = _graph_dir(graph_names)
    run_dir     = _run_dir(graph_names, model_name)
    corpus_path = args.corpus_path or _corpus_path(graph_names)

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    save_graph_plots(graphs, graph_dir)

    if _detect_provider(model_name) == "ollama" and not model_exists(model_name):
        print(f"[ERROR] Ollama model '{model_name}' is not available locally.")
        print("Run 'ollama list' to see installed models, or 'ollama pull <model>' to fetch it.")
        sys.exit(1)

    try:
        corpus = ensure_corpus(graphs, descriptor_names, corpus_path)
        llm    = LLM(model=model_name, run_dir=run_dir)

        df = llm.generate_answers(
            corpus,
            graphs,
            selected_graphs      = graph_names,
            selected_descriptors = descriptor_names,
        )

        eval_plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(eval_plots_dir, exist_ok=True)
        save_eval_plots(df, eval_plots_dir)

        print(f"[DONE] Results    : {llm.save_path}")
        print(f"[DONE] Graph plot : {graph_dir}/graph_*.png")
        print(f"[DONE] Eval plots : {eval_plots_dir}")
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()