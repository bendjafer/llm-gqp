import json
import ast


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_value(val):
    val = str(val).strip()
    try: return ast.literal_eval(val)
    except:
        if val.lower() == "true":  return True
        if val.lower() == "false": return False
        try: return int(val)
        except:
            try: return float(val)
            except: return val


def compare(answer, gt, query_name):
    try:
        a = parse_value(answer)
        g = parse_value(gt)

        if query_name in ("node_count", "edge_count", "degree", "diameter"):
            return int(a) == int(g)

        if query_name in ("clustering", "density"):
            return round(float(a), 2) == round(float(g), 2)

        if query_name in ("is_connected", "cycle"):
            return bool(a) == bool(g)

        if query_name == "neighbors":
            return sorted(str(x) for x in a) == sorted(str(x) for x in g)

        if query_name == "shortest_path":
            al, gl = list(a), list(g)
            return str(al[0]) == str(gl[0]) and str(al[-1]) == str(gl[-1]) and len(al) == len(gl)

    except:
        return False
    return str(a) == str(g)


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(results_path="results/llm_responses.json",
             output_path="results/evaluation.json"):

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    eval_results = {}
    total_correct, total_count, total_errors = 0, 0, 0

    for graph_name, formats in results.items():
        eval_results[graph_name] = {}
        for fmt, queries in formats.items():
            eval_results[graph_name][fmt] = {}
            for query_name, entry in queries.items():
                if "error" in entry:
                    eval_results[graph_name][fmt][query_name] = "ERROR"
                    total_errors += 1
                    continue
                correct = compare(entry["answer"], entry["ground_truth"], query_name)
                eval_results[graph_name][fmt][query_name] = {
                    "correct":      correct,
                    "answer":       entry["answer"],
                    "ground_truth": entry["ground_truth"],
                }
                total_correct += int(correct)
                total_count   += 1

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Correct : {total_correct}/{total_count}  ({100*total_correct/total_count:.1f}%)")
    print(f"Errors  : {total_errors}")
    print(f"\n{'─'*60}")
    print(f"{'Graph':<22} {'Format':<16} {'Query':<20} {'Result'}")
    print(f"{'─'*60}")

    for graph_name, formats in eval_results.items():
        for fmt, queries in formats.items():
            for query_name, res in queries.items():
                if res == "ERROR":
                    status = "ERROR"
                else:
                    status = "CORRECT" if res["correct"] else f"WRONG  got: {res['answer']}  expected: {res['ground_truth']}"
                print(f"{graph_name:<22} {fmt:<16} {query_name:<20} {status}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved in {output_path}")
    return eval_results

if __name__ == "__main__":
    evaluate()