"""E-26: on public SWE-bench agent traces, how far back is the tool output that an edit copies from?

Data: Exgentic `DiscoPosse/agent-llm-traces` parquet shards (SWE-bench rows), plus, as a cross-check,
the user's own Claude Code transcripts (same generalized metric as here, not E-24's Edit-only one).
Writes aggregate numbers only. Needs pyarrow (run with ~/Downloads/agent-llm-traces/.venv).

Predeclared definitions and criteria: RESEARCH_JOURNAL.md, Section 6.1, E-26.
"""

import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics

MIN_LINE = 20
KEEP_LAST = 3
EXCLUDE_SESSION = "dd630f27-8200-40ae-b410-5ae2abcfbc6a"

WRITE_PATTERNS = [
    "sed -i", "perl -pi", "cat >", "cat <<", "tee ", "> /", ">> ", ".write(", ".replace(",
    "patch ", "git apply", "str_replace",
]
OPEN_W = re.compile(r"open\([^)]*['\"]w['\"]")
EDIT_TOOLS = {"Edit", "MultiEdit"}
# Report-only sensitivity variant (added before any statistic was computed): in-place edit forms
# only, dropping the broad redirect and .write/.replace patterns that also match test scripts.
STRICT_PATTERNS = ["sed -i", "perl -pi", "str_replace", "git apply", "patch "]


def is_strict_edit(name, text):
    return name in EDIT_TOOLS or any(p in text for p in STRICT_PATTERNS) or bool(OPEN_W.search(text))


def unescape(text):
    """Undo one level of repr/JSON escaping so escaped file content becomes real lines."""
    return (text.replace("\\\\", "\x00").replace("\\n", "\n").replace("\\t", "\t")
            .replace("\\'", "'").replace('\\"', '"').replace("\x00", "\\"))


def both_forms(text):
    u = unescape(text)
    return text if u == text else text + "\n\x01\n" + u


def candidate_lines(text):
    lines = set()
    for form in (text, unescape(text)):
        for line in form.split("\n"):
            s = line.strip()
            if len(s) >= MIN_LINE:
                lines.add(s)
    return lines


def is_edit(name, text):
    if name in EDIT_TOOLS:
        return True
    return any(p in text for p in WRITE_PATTERNS) or bool(OPEN_W.search(text))


def flatten_values(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in flatten_values(x)]
    if isinstance(v, list):
        return [s for x in v for s in flatten_values(x)]
    return []


def parse_args_text(arguments):
    if isinstance(arguments, (dict, list)):
        return "\n".join(flatten_values(arguments))
    for loader in (ast.literal_eval, json.loads):
        try:
            return "\n".join(flatten_values(loader(arguments)))
        except Exception:
            pass
    return str(arguments)


def decode_result(result):
    """Tool results arrive as a repr of [{'type': 'text', 'text': '{"output": ...}'}] or as JSON."""
    obj = result if not isinstance(result, str) else None
    for loader in (ast.literal_eval, json.loads):
        if obj is not None:
            break
        try:
            obj = loader(result)
        except Exception:
            pass
    if obj is None:
        return result
    texts = []
    items = obj if isinstance(obj, list) else [obj]
    for it in items:
        t = it.get("text") if isinstance(it, dict) and "text" in it else it
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:
                pass
        if isinstance(t, dict):
            texts.extend(str(t.get(k, "")) for k in ("output", "error", "stdout", "stderr") if t.get(k))
            if not any(t.get(k) for k in ("output", "error", "stdout", "stderr")):
                texts.append(json.dumps(t))
        else:
            texts.append(str(t))
    return "\n".join(texts)


def exgentic_events(row):
    """Ordered events of one trace: ('tool', text) and ('action', name, text)."""
    spans = row["spans"]
    counts = [len(json.loads((s["attributes"] or {}).get("gen_ai.input.messages") or "[]")) for s in spans]
    non_monotonic = any(b < a for a, b in zip(counts, counts[1:]))
    big = spans[max(range(len(spans)), key=lambda i: counts[i])]
    msgs = json.loads(big["attributes"]["gen_ai.input.messages"])
    msgs += json.loads(big["attributes"].get("gen_ai.output.messages") or "[]")
    events, seen_task = [], False
    smol = row["harness"] == "smolagents_code"
    for m in msgs:
        role = m.get("role")
        for p in m.get("parts", []):
            t = p.get("type")
            if t == "tool_call":
                events.append(("action", p.get("name"), parse_args_text(p.get("arguments"))))
            elif t == "tool_call_response":
                events.append(("tool", decode_result(p.get("result", ""))))
            elif t == "text" and smol and role == "assistant":
                content = p.get("content", "")
                try:
                    obj, _ = json.JSONDecoder().raw_decode(content[content.index("{"):])
                    code = obj.get("code", "") if isinstance(obj, dict) else content
                except Exception:
                    code = content
                events.append(("action", "code", code))
            elif t == "text" and smol and role == "user":
                if not seen_task:
                    seen_task = True  # the task statement, not a tool output
                    continue
                events.append(("tool", p.get("content", "")))
    return events, non_monotonic


def claude_code_events(path, counted_ids):
    """Main-thread events from one local Claude Code transcript. Tool-use ids already seen in an
    earlier file (forked sessions) stay in history but are not counted again."""
    events = []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("isSidechain") or d.get("type") not in ("assistant", "user"):
                continue
            content = d.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if b.get("type") == "tool_use":
                    dup = b["id"] in counted_ids
                    counted_ids.add(b["id"])
                    events.append(("action", b["name"], "\n".join(flatten_values(b.get("input", {}))), dup))
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    text = c if isinstance(c, str) else "\n".join(
                        x.get("text", "") for x in (c or []) if x.get("type") == "text")
                    events.append(("tool", re.sub(r"^\s*\d+\t", "", text, flags=re.M)))
    return events


def analyse_events(events):
    """Per edit action: is it tool-dependent, and its lag in tool outputs."""
    history = []  # (kind, searchable text, tool index at that point)
    n_tool = 0
    out = []
    for ev in events:
        if ev[0] == "tool":
            history.append(("tool", both_forms(ev[1]), n_tool))
            n_tool += 1
            continue
        name, text = ev[1], ev[2]
        dup = ev[3] if len(ev) > 3 else False
        if is_edit(name, text) and not dup:
            lags = []
            for line in candidate_lines(text):
                src = next((h for h in reversed(history) if line in h[1]), None)
                if src is not None and src[0] == "tool":
                    lags.append(n_tool - 1 - src[2])
            out.append({"name": name, "lags": lags, "strict": is_strict_edit(name, text)})
        history.append(("action", both_forms(text), n_tool))
    return out


def summarise(actions):
    dep = [a for a in actions if a["lags"]]
    max_lags = [max(a["lags"]) for a in dep]
    line_lags = [l for a in dep for l in a["lags"]]
    return {
        "edit_actions": len(actions),
        "tool_dependent": len(dep),
        "tool_dependent_share": len(dep) / max(1, len(actions)),
        "M_share_lag_ge_3": sum(l >= KEEP_LAST for l in max_lags) / max(1, len(max_lags)),
        "line_share_lag_ge_3": sum(l >= KEEP_LAST for l in line_lags) / max(1, len(line_lags)),
        "max_lag_deciles": statistics.quantiles(max_lags, n=10) if len(max_lags) > 10 else max_lags,
        # Report only: share of tool-dependent edits whose source a keep-last-K policy would have
        # cleared, for several K.
        "broken_by_keep_last_k": {k: sum(l >= k for l in max_lags) / max(1, len(max_lags))
                                  for k in (1, 2, 3, 5, 10, 20)},
        "edit_tool_calls": sum(a["name"] in EDIT_TOOLS for a in actions),
        "strict_only": summarise(
            [dict(a, strict=False) for a in actions if a.get("strict")]) if any(
            a.get("strict") for a in actions) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/agent-llm-traces/data"))
    ap.add_argument("--transcripts", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    pooled, by_harness, by_model = [], collections.defaultdict(list), collections.defaultdict(list)
    traces, non_mono, turns = 0, 0, []
    for f in sorted(glob.glob(os.path.join(args.data, "*.parquet"))):
        table = pq.read_table(f, columns=["harness", "benchmark", "models", "spans"])
        for row in table.to_pylist():
            if row["benchmark"] != "swebench":
                continue
            traces += 1
            events, nm = exgentic_events(row)
            non_mono += nm
            turns.append(sum(e[0] == "action" for e in events))
            acts = analyse_events(events)
            pooled += acts
            by_harness[row["harness"]] += acts
            by_model[(row["models"] or ["unknown"])[0]] += acts

    own = []
    counted = set()
    for p in sorted(glob.glob(os.path.join(args.transcripts, "**", "*.jsonl"), recursive=True)):
        if EXCLUDE_SESSION in p or "/subagents/" in p:
            continue
        own += analyse_events(claude_code_events(p, counted))

    pooled_s = summarise(pooled)
    n = pooled_s["tool_dependent"]
    M = pooled_s["M_share_lag_ge_3"]
    verdict = ("INCONCLUSIVE" if n < 100 else "KILL" if M < 0.20 else "SURVIVES" if M >= 0.30 else "PIVOT")
    res = {
        "traces": traces, "traces_non_monotonic_messages": non_mono,
        "actions_per_trace_median": statistics.median(turns) if turns else 0,
        "pooled": pooled_s, "verdict": verdict,
        "by_harness": {k: summarise(v) for k, v in by_harness.items()},
        "by_model": {k: summarise(v) for k, v in by_model.items()},
        "own_transcripts_cross_check": summarise(own),
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
