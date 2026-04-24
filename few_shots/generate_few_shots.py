#!/usr/bin/env python3
"""Generate few-shot examples for LLM-GQP and write them to a JSON file.

Usage:
    python generate_few_shots.py
    python generate_few_shots.py --output few_shots.json --k 3
    python generate_few_shots.py --descriptors adjacency_list edge_list --problems node_count degree
    python generate_few_shots.py --size medium --k 2
"""
import argparse
import json
import os

import networkx as nx

from config import P_PROBLEMS
from descriptors import DESCRIPTORS
from eval import compute_ground_truth
from prompts import get_question, is_valid_problem


_HARDCODED_SMALL: dict[str, nx.Graph] = {
    "triangle":  nx.cycle_graph(3),
    "path5":     nx.path_graph(5),
    "star5":     nx.star_graph(4),    # 5 nodes: center + 4 leaves
    "complete4": nx.complete_graph(4),
    "cycle6":    nx.cycle_graph(6),
}


def _get_graphs(size: str) -> dict[str, nx.Graph]:
    if size == "small":
        return dict(_HARDCODED_SMALL)
    # medium / large: supplement with 3 generated random graphs
    from graphs import generate_graphs
    graphs = dict(_HARDCODED_SMALL)
    for seed in range(3):
        generated = generate_graphs(size=size, seed=seed)
        graphs.update(generated)
    print(f"[INFO] --size={size}: using {len(graphs)} graphs "
          f"(5 hardcoded small + {len(graphs) - 5} generated)")
    return graphs


def _fmt_answer(gt) -> str:
    return f"<answer>{gt}</answer>"


def generate_examples(
    graphs: dict,
    descriptor_names: list[str],
    problem_names: list[str],
    k: int,
) -> list[dict]:
    """Return up to k examples per (problem, descriptor) pair.

    Skips:
    - invalid problem/graph combos (e.g. diameter on disconnected graph)
    - combos where ground truth is None
    """
    seen: dict[tuple, int] = {}          # (problem, descriptor) -> count
    seen_desc: set[tuple] = set()         # (problem, descriptor, graph_description) — dedup
    examples: list[dict] = []

    for graph_name, G in graphs.items():
        directed = G.is_directed()
        for problem in problem_names:
            if not is_valid_problem(problem, G):
                continue
            gt = compute_ground_truth(G, problem)
            if gt is None:
                continue
            question, format_rule = get_question(problem, directed)

            for descriptor in descriptor_names:
                key = (problem, descriptor)
                if seen.get(key, 0) >= k:
                    continue
                try:
                    graph_description = DESCRIPTORS[descriptor](G)
                except Exception:
                    continue

                desc_key = (problem, descriptor, graph_description)
                if desc_key in seen_desc:
                    continue
                seen_desc.add(desc_key)

                examples.append({
                    "problem":           problem,
                    "descriptor":        descriptor,
                    "format_label":      descriptor,
                    "graph_description": graph_description,
                    "question":          question,
                    "format_rule":       format_rule,
                    "final_answer_xml":  _fmt_answer(gt),
                })
                seen[key] = seen.get(key, 0) + 1

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples for LLM-GQP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", default="few_shots.json",
        help="Output JSON file (default: few_shots.json)",
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="small",
        help="Graph size: 'small' uses 5 hardcoded graphs; "
             "'medium'/'large' also adds generated graphs (default: small)",
    )
    parser.add_argument(
        "--descriptors", nargs="+", default=sorted(DESCRIPTORS),
        choices=sorted(DESCRIPTORS), metavar="DESCRIPTOR",
        help="Descriptor formats to include (default: all). "
             f"Available: {sorted(DESCRIPTORS)}",
    )
    parser.add_argument(
        "--problems", nargs="+", default=list(P_PROBLEMS),
        choices=list(P_PROBLEMS), metavar="PROBLEM",
        help=f"Problems to include (default: all). Available: {list(P_PROBLEMS)}",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Max examples per (problem, descriptor) combo (default: 3, max useful: 3)",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")

    graphs   = _get_graphs(args.size)
    examples = generate_examples(graphs, args.descriptors, args.problems, args.k)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=True)

    n_combos = len(set((e["problem"], e["descriptor"]) for e in examples))
    print(f"[INFO] {len(examples)} examples across {n_combos} "
          f"(problem × descriptor) combos → {args.output}")


if __name__ == "__main__":
    main()
