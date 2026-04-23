import re

import networkx as nx

from config import P_PROBLEMS


ROLE_AND_RULES = """\
You are an expert graph theorist. You analyze {graph_description} precisely.

Respond with ONLY this XML tag:
<answer>
[your final answer only — no labels, no explanation, no reasoning]
</answer>

Rules:
- Never leave <answer> empty.
- Use the graph exactly as given. Do not invent edges or nodes.
- {graph_note}
{directed_notes}"""

_DIRECTED_NOTES = """\
- This graph is DIRECTED. Every edge is one-way.
- "Neighbors" = out-neighbors (successors) only.
- "Degree" = in-degree + out-degree.
- "Connected" = weakly connected (reachable ignoring direction).
- "Cycle" = directed cycle."""

_UNDIRECTED_NOTES = """\
- This graph is UNDIRECTED. Every edge is bidirectional. Count each edge once."""

PROMPT_TEMPLATE = """\
Graph (format: {format_label}):
{description}

Question: {question}
Answer format: {format_rule}"""


def build_prompt_vars(G) -> dict:
    """ROLE_AND_RULES template variables derived from the actual NetworkX graph."""
    directed = G.is_directed() if G is not None else False
    weighted = bool(nx.get_edge_attributes(G, "weight")) if G is not None else False
    return {
        "graph_description": f"{'directed' if directed else 'undirected'}, {'weighted' if weighted else 'unweighted'} graphs",
        "graph_note":        "Treat every edge as one-way." if directed else "Every edge is bidirectional.",
        "directed_notes":    _DIRECTED_NOTES if directed else _UNDIRECTED_NOTES,
    }


def get_question(problem: str, directed: bool) -> tuple[str, str]:
    """Return (question_text, format_rule) adapted for graph direction."""
    spec = P_PROBLEMS[problem]
    q    = spec.directed_question if (directed and spec.directed_question) else spec.question
    return q, spec.format_rule


def is_valid_problem(problem: str, G) -> bool:
    """Return False when the problem is undefined for this graph.

    Currently: diameter is skipped for graphs that are disconnected
    (undirected) or not strongly connected (directed), since there is
    no finite answer and the LLM call would be wasted.
    """
    if G is None or problem != "diameter":
        return True
    return nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)


# Compiled once at module load.
_MD_MARKERS    = re.compile(r"[*_`]+")
_PROSE_PREFIX  = re.compile(
    r"^(?:the\s+)?(?:answer|result|value)\s+is\s*[:\-]?\s*",
    re.IGNORECASE,
)
# Matches the most specific embedded value type, ordered: list > float > int > bool.
_EXTRACT_VALUE = re.compile(
    r"(\[[\d,\s\-\.]+\]|True|False|\-?\d+\.\d+|\-?\d+)",
    re.IGNORECASE,
)


def _clean_answer(text: str) -> str:
    """Strip LLM prose and markdown from the content of an <answer> block.

    Handles:
    - Markdown bold/italic/code markers (**x**, *x*, `x`)
    - Prose prefixes: "The answer is:", "Result:", "Answer:", etc.
    - Domain prose: "The clustering coefficient is 0.3333" → "0.3333"
    - Multi-line answers: takes the first non-empty line
    - Trailing sentence punctuation
    """
    text  = _MD_MARKERS.sub("", text.strip())
    text  = _PROSE_PREFIX.sub("", text).strip()
    first = next((l.strip() for l in text.splitlines() if l.strip()), text)
    first = first.rstrip(".,;:")
    # If the line is still prose (>2 words), try to extract the embedded value.
    if len(first.split()) > 2:
        m = _EXTRACT_VALUE.search(first)
        if m:
            return m.group(1)
    return first


def parse_response(text: str) -> str:
    """Extract the answer from an <answer>...</answer> block.

    Falls back to the full response text if the tag is missing.
    Warns when output is malformed.
    """
    a = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)

    if not a:
        print("[WARN] LLM output missing <answer> tag — using full response as fallback")
        return _clean_answer(text)

    raw    = a.group(1).strip()
    answer = _clean_answer(raw)
    if answer != raw:
        print(f"[WARN] Answer prose stripped: {repr(raw)[:80]} → {repr(answer)}")
    return answer