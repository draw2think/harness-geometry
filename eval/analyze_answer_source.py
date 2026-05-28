"""
Answer Source Taxonomy — classify how each problem's answer was derived.

Categories:
  clean_oracle       : query value directly matches the final answer
  hybrid             : query/CAS returned an intermediate value, LLM computed the final answer from it
  resilient_oracle   : tool failures occurred, the model recovered, and the final answer still traces to an engine value
  resilient_fallback : tool failures occurred, but the final answer came from reasoning rather than an engine value
  llm_bypass         : tools were called but the answer doesn't trace to any engine-returned value
  direct_reasoning   : no tools called at all (fell back to pure VLM reasoning)

Usage:
    python eval/analyze_answer_source.py --dataset pgps9k --model gemini-3-flash-preview
    python eval/analyze_answer_source.py --dir eval/pgps9k  # scan all results in directory
"""
import json
import glob
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def close_match(answer_val, query_val, choices=None, tol=0.5):
    """Check if answer matches a query value, directly or via choice mapping."""
    if answer_val is None or query_val is None:
        return False

    # Direct numeric match
    try:
        a = float(answer_val)
        q = float(query_val)
        if abs(a - q) < tol:
            return True
    except (ValueError, TypeError):
        pass

    # String match (letter answers)
    if str(answer_val).strip().upper() == str(query_val).strip().upper():
        return True

    # Choice mapping: query returns 313.0, answer is "D", choices[3]=313
    if choices and isinstance(choices, list):
        try:
            q = float(query_val)
            for i, c in enumerate(choices):
                try:
                    if abs(float(c) - q) < tol:
                        letter = chr(65 + i)  # A, B, C, D
                        if str(answer_val).strip().upper() == letter:
                            return True
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            pass

    return False


def classify(result):
    """Classify a single result's answer source."""
    process = result.get("process", {})
    turns = process.get("turns", [])
    answer_raw = result.get("answer_raw")
    passed = result.get("passed", False)
    choices = result.get("choices", [])

    if not answer_raw:
        return "no_answer"

    answer_val = answer_raw.get("value")

    # Collect all tool calls info
    all_query_values = []
    has_tool_failure = False
    total_tools = 0
    total_fails = 0
    tool_fns = []

    for turn in turns:
        for tc in turn.get("tool_calls", []):
            total_tools += 1
            tool_fns.append(tc.get("fn", ""))
            if not tc.get("ok", False):
                has_tool_failure = True
                total_fails += 1
            if tc.get("ok") and tc.get("value") is not None:
                all_query_values.append({
                    "fn": tc.get("fn", ""),
                    "value": tc["value"],
                    "cmd": tc.get("cmd", ""),
                })

    # No tools at all
    if total_tools == 0:
        return "direct_reasoning"

    # Check if any query value matches the answer
    matches_query = False
    matching_query = None
    for qv in all_query_values:
        if close_match(answer_val, qv["value"], choices):
            matches_query = True
            matching_query = qv
            break

    # Classification (priority order)
    if has_tool_failure and passed:
        # Had failures but still got it right
        if matches_query:
            return "resilient_oracle"  # recovered AND used engine value
        else:
            return "resilient_fallback"  # recovered via reasoning

    if matches_query:
        return "clean_oracle"

    if all_query_values and not matches_query:
        # Had query values but answer doesn't directly match
        # Check if it could be derived (e.g., Solve→x=94, answer=63)
        return "hybrid"

    if not all_query_values and total_tools > 0:
        # Used construction tools but never queried
        return "llm_bypass"

    return "unknown"


def analyze_results(result_dir, model_filter=None, id_min=None, id_max=None):
    """Analyze all result.json files in a directory.

    id_min/id_max: optional integer range to filter problem IDs (inclusive).
    Useful for split selection, e.g., Geometry3K test split: id_min=2401, id_max=3001.
    """
    pattern = str(Path(result_dir) / "*" / "*_result.json")
    files = glob.glob(pattern)

    if model_filter:
        files = [f for f in files if model_filter in f]

    if id_min is not None or id_max is not None:
        def _in_range(f):
            try:
                pid = int(Path(f).parent.name)
                return (id_min is None or pid >= id_min) and (id_max is None or pid <= id_max)
            except ValueError:
                return True  # non-numeric IDs pass through
        files = [f for f in files if _in_range(f)]

    if not files:
        print(f"No result files found in {result_dir}")
        return

    # Classify each result
    classifications = []
    details = []

    for fpath in sorted(files):
        try:
            result = json.load(open(fpath))
        except Exception:
            continue

        cat = classify(result)
        passed = result.get("passed", False)
        prob_id = result.get("id", Path(fpath).parent.name)
        model = result.get("model", "unknown")

        classifications.append(cat)
        details.append({
            "id": prob_id,
            "model": model,
            "category": cat,
            "passed": passed,
            "answer": (result.get("answer_raw") or {}).get("value"),
            "turns": result.get("process", {}).get("total_turns", 0),
            "file": fpath,
        })

    # Summary
    counter = Counter(classifications)
    total = len(classifications)

    print(f"\n{'=' * 60}")
    print(f"  Answer Source Taxonomy — {total} problems")
    print(f"  Directory: {result_dir}")
    if model_filter:
        print(f"  Model filter: {model_filter}")
    print(f"{'=' * 60}\n")

    # Category breakdown
    category_order = [
        "clean_oracle", "hybrid", "resilient_oracle",
        "resilient_fallback", "llm_bypass", "direct_reasoning",
        "no_answer", "unknown"
    ]
    category_desc = {
        "clean_oracle": "Query value = answer (engine-grounded)",
        "hybrid": "Query/CAS intermediate → LLM computed final",
        "resilient_oracle": "Tool fail → recovered → used engine value",
        "resilient_fallback": "Tool fail → recovered via reasoning",
        "llm_bypass": "Tools called but answer not from query",
        "direct_reasoning": "No tools called (pure VLM)",
        "no_answer": "No answer emitted",
        "unknown": "Unclassified",
    }

    print(f"  {'Category':<25s} {'Count':>6s} {'%':>7s}  Description")
    print(f"  {'-' * 75}")
    for cat in category_order:
        count = counter.get(cat, 0)
        if count > 0:
            pct = count / total * 100
            desc = category_desc.get(cat, "")
            marker = "★" if cat == "clean_oracle" else " "
            print(f"  {marker} {cat:<23s} {count:>5d} {pct:>6.1f}%  {desc}")

    # Pass rate by category
    print(f"\n  {'Category':<25s} {'Pass':>5s} {'Total':>6s} {'Rate':>7s}")
    print(f"  {'-' * 50}")
    for cat in category_order:
        items = [d for d in details if d["category"] == cat]
        if items:
            passed = sum(1 for d in items if d["passed"])
            print(f"  {cat:<25s} {passed:>5d} {len(items):>6d} {passed/len(items)*100:>6.1f}%")

    # Aggregate ratios used in the paper.
    # Engine-direct = answer directly matches an engine-returned value.
    engine_direct = counter.get("clean_oracle", 0) + counter.get("resilient_oracle", 0)
    # Engine-involved = answer path uses at least one engine-returned value, even if the
    # model performs a final derivation step on top of that value.
    engine_involved = engine_direct + counter.get("hybrid", 0)

    print(
        f"\n  Engine-Direct Ratio: {engine_direct}/{total} = "
        f"{engine_direct/total*100:.1f}% of answers directly match an engine value"
    )
    print(
        f"  Engine-Involved Ratio: {engine_involved}/{total} = "
        f"{engine_involved/total*100:.1f}% of answers use at least one engine-returned value"
    )

    # Per-problem details (show interesting cases)
    bypass_cases = [d for d in details if d["category"] == "llm_bypass"]
    if bypass_cases:
        print(f"\n  LLM Bypass cases (tools called but answer not from engine):")
        for d in bypass_cases[:10]:
            print(f"    {d['id']:>12s}  answer={d['answer']}  turns={d['turns']}")

    resilient_cases = [d for d in details if d["category"].startswith("resilient")]
    if resilient_cases:
        print(f"\n  Resilient cases (recovered from tool failures):")
        for d in resilient_cases[:10]:
            print(f"    {d['id']:>12s}  answer={d['answer']}  passed={d['passed']}  cat={d['category']}")

    return details


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Answer Source Taxonomy Analysis")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing result.json files")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., pgps9k)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model name substring")
    parser.add_argument("--id-min", type=int, default=None,
                        help="Minimum problem ID (inclusive), e.g. 2401 for Geo3K test split")
    parser.add_argument("--id-max", type=int, default=None,
                        help="Maximum problem ID (inclusive), e.g. 3001 for Geo3K test split")
    args = parser.parse_args()

    if args.dir:
        result_dir = args.dir
    elif args.dataset:
        result_dir = f"eval/{args.dataset}"
    else:
        result_dir = "eval/pgps9k"

    analyze_results(result_dir, model_filter=args.model,
                    id_min=args.id_min, id_max=args.id_max)
