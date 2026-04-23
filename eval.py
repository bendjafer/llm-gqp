import ast
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


FLOAT_TOLERANCE = 1e-2  # raised from 1e-3: LLMs round to 4 decimal places


def compute_ground_truth(G, problem):
    """Compute the correct answer for a NetworkX graph G and problem name."""
    nodes = list(G.nodes())
    first_node = nodes[0]
    last_node  = nodes[-1]
    directed   = G.is_directed()

    match problem:
        case "node_count":
            return G.number_of_nodes()
        case "edge_count":
            return G.number_of_edges()
        case "neighbors":
            nbrs = list(G.successors(first_node)) if directed else list(G.neighbors(first_node))
            return sorted(nbrs, key=str)
        case "degree":
            return G.degree(first_node)
        case "shortest_path":
            if not nx.has_path(G, first_node, last_node):
                return None
            return nx.shortest_path(G, first_node, last_node)
        case "clustering":
            return round(nx.clustering(G, first_node), 4)
        case "diameter":
            gate = nx.is_strongly_connected(G) if directed else nx.is_connected(G)
            if not gate:
                return None
            return nx.diameter(G)
        case "density":
            return round(nx.density(G), 4)
        case "is_connected":
            return nx.is_weakly_connected(G) if directed else nx.is_connected(G)
        case "cycle":
            if directed:
                return any(nx.simple_cycles(G))  # short-circuits at first cycle found
            return len(nx.cycle_basis(G)) > 0
        case _:
            return None


def add_ground_truth(df, graphs, save_path=None):
    """Add a ground_truth column to df; optionally save CSV."""
    def get_gt(row):
        graph = graphs.get(row["graph_name"])
        if graph is None:
            print(f"[WARN] Graph '{row['graph_name']}' not found")
            return None
        return compute_ground_truth(graph, row["problem"])

    df = df.copy()
    df["ground_truth"] = df.apply(get_gt, axis=1)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved → {save_path}")

    return df


def parse_answer(val):
    """Parse an LLM answer string into the most specific Python type possible.

    Handles:
    - Already-typed values (bool, int, float, list, …) returned as-is.
    - NaN / None → None so the row is counted as unparseable.
    - Strings that are valid Python literals (lists, booleans, numbers).
    - Plain strings that cannot be parsed further → returned unchanged.
    """
    if val is None:
        return None
    # Already a native Python type — no parsing needed.
    if isinstance(val, (bool, int, float, list, dict, tuple)):
        return val
    # Catch pandas NaN / numpy NaN stored as float.
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    text = str(val).strip()

    # Treat explicit sentinel strings written by the LLM or the pipeline as failures.
    if text.lower() in ("no answer found", "none", "nan", ""):
        return None

    # Try Python literal evaluation first (handles lists, booleans, numbers).
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    # Case-insensitive boolean fallback for plain "true"/"false" strings.
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False

    # Return the raw string so the caller can still attempt a string comparison.
    return text


def _answers_match(answer, ground_truth, problem):
    """Return True/False/None comparing a parsed LLM answer to the ground truth.

    Returns None when either value is absent so the row can be excluded from
    accuracy but counted toward the unparseable rate.
    All branches return a plain Python bool (not a numpy bool) so the result
    is safe to store in a pandas column and compare with == True / == False.
    """
    if answer is None or ground_truth is None:
        return None

    match problem:
        case "node_count" | "edge_count" | "diameter" | "degree":
            try:
                return bool(int(answer) == int(ground_truth))
            except (TypeError, ValueError):
                return False

        case "is_connected" | "cycle":
            # The ground truth is always a Python bool.
            # The LLM answer may arrive as bool, or as a string like "True".
            if isinstance(answer, bool):
                return bool(answer == ground_truth)
            if isinstance(answer, str):
                return bool(answer.strip().lower() == str(ground_truth).lower())
            # Numeric fallback: treat non-zero as True.
            try:
                return bool(bool(answer) == bool(ground_truth))
            except (TypeError, ValueError):
                return False

        case "density" | "clustering":
            try:
                # Round both sides to 4 d.p. before comparing so LLM rounding
                # conventions don't count as errors.
                return bool(
                    abs(round(float(answer), 4) - round(float(ground_truth), 4))
                    <= FLOAT_TOLERANCE
                )
            except (TypeError, ValueError):
                return False

        case "neighbors":
            # Both must be lists; compare sorted string representations so node
            # identifier types (int vs str) don't cause spurious mismatches.
            if not isinstance(answer, list) or not isinstance(ground_truth, list):
                return False
            return bool(
                sorted(str(x) for x in answer) == sorted(str(x) for x in ground_truth)
            )

        case "shortest_path":
            if not isinstance(answer, list) or not isinstance(ground_truth, list):
                return False
            # Compare element-by-element as strings to tolerate int/str node id mixing.
            return bool(
                len(answer) == len(ground_truth)
                and all(str(a) == str(g) for a, g in zip(answer, ground_truth))
            )

        case _:
            return bool(answer == ground_truth)


def _safe_match(answer, ground_truth, problem):
    """Wrapper around _answers_match that returns None instead of raising.

    Prevents a single malformed row from aborting the entire DataFrame apply.
    """
    try:
        return _answers_match(answer, ground_truth, problem)
    except Exception:
        return None


def add_correctness(df):
    """Add answer_parsed and correct columns."""
    df = df.copy()
    df["answer_parsed"] = df["answer"].apply(parse_answer)
    df["correct"] = df.apply(
        lambda row: _safe_match(
            row["answer_parsed"],
            row["ground_truth"],
            row["problem"],
        ),
        axis=1,
    )
    return df


def accuracy_by(df, col):
    """Return accuracy (%) per unique value of `col`."""
    grouped = df.groupby(col)["correct"]
    return (grouped.sum() / grouped.count() * 100).round(2).rename("accuracy_%")


def parse_rate(df):
    """Fraction of rows where the answer was successfully parsed."""
    parsed = df["answer_parsed"].notna()
    total = len(df)
    count = parsed.sum()
    print(f"Parse rate: {count}/{total} ({100 * count / total:.1f}%)")
    return parsed.mean()


def full_report(df):
    """Print a comprehensive text report to stdout."""
    n_total = len(df)
    n_correct = df["correct"].sum()
    n_none = df["correct"].isna().sum()

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total rows : {n_total}")
    print(f"Correct answers : {int(n_correct)}")
    print(f"Unparseable answers : {int(n_none)}")
    scoreable = n_total - n_none
    overall = f"{100 * n_correct / scoreable:.2f}%" if scoreable else "N/A (all unparseable)"
    print(f"Overall accuracy : {overall} (excl. unparseable)")
    print()

    for dim, label in [
        ("problem", "BY PROBLEM"),
        ("format", "BY FORMAT"),
        ("graph_name", "BY GRAPH"),
    ]:
        acc = accuracy_by(df.dropna(subset=["correct"]), dim)
        print(f"── {label} " + "─" * (50 - len(label)))
        print(acc.to_string())
        print()

    print("── ERROR EXAMPLES " + "─" * 42)
    errors = df[df["correct"] == False][
        ["graph_name", "format", "problem", "answer_parsed", "ground_truth"]
    ].head(10)
    print(errors.to_string(index=False))
    print("=" * 60)


def plot_accuracy_bar(df, col, title=None, ax=None, color="steelblue"):
    """Horizontal bar chart: accuracy (%) per value of `col`."""
    acc = accuracy_by(df.dropna(subset=["correct"]), col).sort_values()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    acc.plot.barh(ax=ax, color=color, edgecolor="white")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title(title or f"Accuracy by {col}")

    for bar in ax.patches:
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%",
            va="center",
            fontsize=9,
        )
    return ax


def plot_heatmap(df, row_col, col_col, title=None, ax=None, fmt=".0f"):
    """Heatmap of accuracy (%) for every (row_col × col_col) combination."""
    sub = df.dropna(subset=["correct"])
    pivot = (
        sub.groupby([row_col, col_col])["correct"]
        .apply(lambda s: 100 * s.sum() / len(s))
        .unstack(col_col)
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, len(pivot.columns) * 1.1), max(4, len(pivot.index) * 0.7))
        )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
    )
    ax.set_title(title or f"Accuracy: {row_col} × {col_col}")
    ax.set_ylabel(row_col)
    ax.set_xlabel(col_col)
    plt.tight_layout()
    return ax


def plot_error_distribution(df, ax=None):
    """Stacked bar: correct / wrong / unparseable per problem."""
    counts = df.groupby("problem")["correct"].value_counts(dropna=False).unstack(fill_value=0)
    counts = counts.rename(columns={True: "correct", False: "wrong", None: "unparseable"})

    for col in ["correct", "wrong", "unparseable"]:
        if col not in counts.columns:
            counts[col] = 0

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    counts[["correct", "wrong", "unparseable"]].plot.bar(
        stacked=True,
        ax=ax,
        color=["#4caf50", "#f44336", "#9e9e9e"],
        edgecolor="white",
    )
    ax.set_title("Answer distribution per problem")
    ax.set_xlabel("Problem")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return ax


def plot_confusion_per_problem(df, problem, ax=None):
    """Scatter plot of ground truth vs parsed answer for one numerical problem."""
    sub = df[df["problem"] == problem].dropna(subset=["answer_parsed", "ground_truth"])

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    colors = sub["correct"].map({True: "#4caf50", False: "#f44336", None: "#9e9e9e"})

    try:
        gt = sub["ground_truth"].astype(float)
        ap = sub["answer_parsed"].astype(float)
        ax.scatter(gt, ap, c=colors, alpha=0.7, edgecolors="k", linewidths=0.4)
        limits = [min(gt.min(), ap.min()), max(gt.max(), ap.max())]
        ax.plot(limits, limits, "k--", alpha=0.4, label="perfect")
    except (TypeError, ValueError):
        ax.text(0.5, 0.5, "non-numeric", ha="center", transform=ax.transAxes)

    ax.set_title(f"Predicted vs True — {problem}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("LLM answer")
    return ax


def save_eval_plots(df, plots_dir):
    """Generate and save all standard evaluation plots to plots_dir."""
    os.makedirs(plots_dir, exist_ok=True)  # os imported at module level

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_accuracy_bar(df, "problem",    "Accuracy by Problem", ax=axes[0], color="#5c85d6")
    plot_accuracy_bar(df, "format",     "Accuracy by Format",  ax=axes[1], color="#d67c5c")
    plot_accuracy_bar(df, "graph_name", "Accuracy by Graph",   ax=axes[2], color="#5cb85c")
    plt.suptitle("LLM Graph Understanding — Accuracy Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_bars.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_heatmap(df, row_col="problem", col_col="format",
                 title="Accuracy (%) — Problem × Format", ax=ax)
    plt.savefig(os.path.join(plots_dir, "heatmap_problem_format.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_error_distribution(df, ax=ax)
    plt.savefig(os.path.join(plots_dir, "error_distribution.png"), dpi=150)
    plt.close()

    print(f"[INFO] Plots saved → {plots_dir}")