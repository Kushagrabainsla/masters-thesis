"""E-24: what breaks the prompt cache in a real harness, and the oracle ceiling for
compress-on-ingest of tool results (idea I-41).

Reads the user's local Claude Code transcripts (read-only) and writes aggregate numbers only.
Raw transcript text never leaves the machine and is not written into the repo. Samples for
E-25 (which contain code) go to a scratch directory passed with --samples-out.

Predeclared definitions and criteria: RESEARCH_JOURNAL.md, Section 6.1, E-24.
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
from datetime import datetime

EXCLUDE_SESSION = "dd630f27-8200-40ae-b410-5ae2abcfbc6a"  # the session that wrote this script

# List prices per MTok (input, output) and cache-read multiplier, from the Claude
# prompt-caching docs page, checked 2026-10-04.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.05),
    "claude-opus-5": (5.0, 25.0, 0.10),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-sonnet-5": (2.0, 10.0, 0.10),
    "claude-haiku-4-5-20251001": (1.0, 5.0, 0.10),
}
WRITE_5M, WRITE_1H = 1.25, 2.0
HAIKU_IN, HAIKU_OUT = 1.0, 5.0
COMPRESSOR_PROMPT_TOKENS = 300
INGEST_THRESHOLD = 1024
CHARS_PER_OUTPUT_TOKEN = 4
IMAGE_CHARS = 6000  # an image block counted as about 1,500 tokens

# Attachments that log the prefix (system prompt, tool schemas) rather than appended messages.
PREFIX_ATTACHMENTS = {"prompt_snapshot", "deferred_tools_record"}

READ_PREFIX = re.compile(r"^\s*\d+\t", re.M)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def block_text(content):
    """Text and a char-equivalent size of a tool_result content field."""
    if isinstance(content, str):
        return content, len(content)
    text, size = [], 0
    for b in content or []:
        if b.get("type") == "text":
            text.append(b.get("text", ""))
            size += len(b.get("text", ""))
        elif b.get("type") == "image":
            size += IMAGE_CHARS
    return "\n".join(text), size


def load_stream(path, sidechain):
    """Turn one transcript into an ordered list of API requests for one thread.

    Each request carries its final usage plus the content appended since the previous
    request: tool results (with tool name and text), other characters, and compaction flags.
    """
    records = []
    with open(path) as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    tool_names = {}
    last_usage = {}
    for d in records:
        if d.get("type") == "assistant" and d.get("requestId"):
            msg = d.get("message", {})
            if "usage" in msg:
                last_usage[d["requestId"]] = (msg["usage"], msg.get("model"))
            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_names[b["id"]] = b["name"]

    requests, seen = [], set()
    pending = {"tool": [], "other_chars": 0, "compactions": []}
    for d in records:
        if bool(d.get("isSidechain")) != sidechain:
            continue
        t = d.get("type")
        if t == "system" and d.get("subtype") == "compact_boundary":
            pending["compactions"].append(d.get("compactMetadata") or {})
        elif t == "attachment":
            a = d.get("attachment")
            if isinstance(a, dict) and a.get("type") not in PREFIX_ATTACHMENTS:
                pending["other_chars"] += len(json.dumps(a))
        elif t == "user":
            c = d.get("message", {}).get("content")
            if isinstance(c, str):
                pending["other_chars"] += len(c)
            elif isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result":
                        text, size = block_text(b.get("content"))
                        name = tool_names.get(b.get("tool_use_id"), "unknown")
                        pending["tool"].append({"name": name, "text": text, "chars": size})
                    elif b.get("type") == "text":
                        pending["other_chars"] += len(b.get("text", ""))
                    elif b.get("type") == "image":
                        pending["other_chars"] += IMAGE_CHARS
        elif t == "assistant":
            rid = d.get("requestId")
            if not rid or rid in seen or rid not in last_usage:
                continue
            usage, model = last_usage[rid]
            if model not in PRICES:
                continue
            seen.add(rid)
            requests.append({
                "rid": rid, "model": model, "ts": ts(d["timestamp"]), "usage": usage,
                "new_tool": pending["tool"], "new_other_chars": pending["other_chars"],
                "compactions": pending["compactions"],
            })
            pending = {"tool": [], "other_chars": 0, "compactions": []}

    # Edits issued by each request, for the verbatim-dependency check.
    by_rid = {r["rid"]: r for r in requests}
    for r in requests:
        r["edits"] = []
        r["self_texts"] = []  # text this request wrote itself (Write content, Edit new_string)
    for d in records:
        if d.get("type") != "assistant" or bool(d.get("isSidechain")) != sidechain:
            continue
        r = by_rid.get(d.get("requestId"))
        if r is None:
            continue
        for b in d.get("message", {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b["name"] in ("Edit", "MultiEdit"):
                inp = b.get("input", {})
                olds = [inp.get("old_string")] if b["name"] == "Edit" else [
                    e.get("old_string") for e in inp.get("edits", [])]
                for o in olds:
                    if isinstance(o, str) and len(o) >= 20:
                        r["edits"].append(o)
            if isinstance(b, dict) and b.get("type") == "tool_use":
                inp = b.get("input", {})
                if b["name"] == "Write":
                    r["self_texts"].append(inp.get("content") or "")
                elif b["name"] == "Edit":
                    r["self_texts"].append(inp.get("new_string") or "")
                elif b["name"] == "MultiEdit":
                    r["self_texts"].extend(e.get("new_string") or "" for e in inp.get("edits", []))
    return requests


def request_prices(r):
    base_in, base_out, read_mult = PRICES[r["model"]]
    u = r["usage"]
    cc = u.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens", 0) or 0
    w5m = cc.get("ephemeral_5m_input_tokens", 0) or 0
    write_total = u.get("cache_creation_input_tokens", 0) or 0
    if w1h + w5m == 0 and write_total:
        w5m = write_total
    write_mult = (w1h * WRITE_1H + w5m * WRITE_5M) / max(1, w1h + w5m) if write_total else WRITE_5M
    return {
        "in": base_in / 1e6, "out": base_out / 1e6, "read": base_in * read_mult / 1e6,
        "write": base_in * write_mult / 1e6, "ttl_1h": w1h > 0,
    }


def analyse(streams, ratios=(2, 4, 10)):
    agg = collections.Counter()
    cost = collections.Counter()
    break_tokens = collections.Counter()
    break_counts = collections.Counter()
    break_cost = collections.Counter()
    tool_tokens_by_name = collections.Counter()
    oracle_saving = collections.Counter()
    realistic = collections.Counter()
    compactions = []
    capped = 0
    gaps_min = []
    tpc_samples = []
    cost_compaction_call_est = [0.0]

    for reqs in streams:
        prev = None
        tt = 0.0   # all tool-result tokens currently in context (oracle)
        ct = 0.0   # tool-result tokens from results >= threshold (realistic)
        for r in reqs:
            u = r["usage"]
            inp = u.get("input_tokens", 0) or 0
            cw = u.get("cache_creation_input_tokens", 0) or 0
            cr = u.get("cache_read_input_tokens", 0) or 0
            out = u.get("output_tokens", 0) or 0
            p = request_prices(r)
            # w = 0 for a request already counted from an earlier file (forked or resumed
            # session). It still updates the running state below but adds nothing to totals.
            w = 0 if r["dup"] else 1
            cost["uncached_input"] += w * inp * p["in"]
            cost["cache_write"] += w * cw * p["write"]
            cost["cache_read"] += w * cr * p["read"]
            cost["output"] += w * out * p["out"]
            agg["requests"] += w
            agg["input_tokens"] += w * inp
            agg["cache_write_tokens"] += w * cw
            agg["cache_read_tokens"] += w * cr
            agg["output_tokens"] += w * out

            # Cache-break classification.
            P_prev = None
            if prev is not None:
                pu = prev["usage"]
                P_prev = sum(pu.get(k, 0) or 0 for k in (
                    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                if w:
                    gaps_min.append((r["ts"] - prev["ts"]) / 60)
            if prev is None:
                cls = "session_start"
            elif cr >= 0.9 * P_prev:
                cls = "hit"
            elif r["compactions"]:
                cls = "compaction"
            elif r["model"] != prev["model"]:
                cls = "model_switch"
            elif (r["ts"] - prev["ts"]) > (3600 if request_prices(prev)["ttl_1h"] else 300):
                cls = "ttl_expiry"
            else:
                cls = "other_break"
            break_counts[cls] += w
            break_tokens[cls] += w * (cw + inp)
            break_cost[cls] += w * (cw * p["write"] + inp * p["in"])

            for meta in (r["compactions"] if w else []):
                # The summarization call is not logged as a request. Estimate it as one read of
                # the pre-compaction context plus the summary as output, at this model's prices.
                est = (meta.get("preTokens") or 0) * p["read"] + (meta.get("postTokens") or 0) * p["out"]
                cost_compaction_call_est[0] += est
                compactions.append({"pre": meta.get("preTokens"), "post": meta.get("postTokens"),
                                    "duration_s": (meta.get("durationMs") or 0) / 1000,
                                    "trigger": meta.get("trigger"), "tool_tokens_in_context": tt})
            if r["compactions"]:
                tt = ct = 0.0

            # Attribute appended tokens to tool results by character share. The first request
            # carries the system prompt and a compaction request carries the summary, neither
            # of which is logged as characters, so neither gets a tool attribution.
            tool_chars = sum(x["chars"] for x in r["new_tool"])
            prev_out_chars = (prev["usage"].get("output_tokens", 0) or 0) * CHARS_PER_OUTPUT_TOKEN if prev else 0
            denom = tool_chars + r["new_other_chars"] + prev_out_chars
            if cls in ("session_start", "compaction"):
                appended = 0
            else:
                appended = max(0, (inp + cw + cr) - P_prev)
            tok_per_char = (appended / denom) if denom else 0.0
            if w and denom and tool_chars:
                tpc_samples.append(tok_per_char)
            new_tool = tool_chars * tok_per_char
            new_ct = sum(x["chars"] for x in r["new_tool"]
                         if x["chars"] * tok_per_char >= INGEST_THRESHOLD) * tok_per_char
            for x in r["new_tool"]:
                tool_tokens_by_name[x["name"]] += w * x["chars"] * tok_per_char
            agg["appended_tokens"] += w * appended
            agg["appended_tool_tokens"] += w * new_tool

            # Where the tool tokens of this request were billed.
            if cls == "hit":
                read_tool, write_tool = tt, new_tool
                read_ct, write_ct = ct, new_ct
            else:
                read_tool, write_tool = 0.0, tt + new_tool
                read_ct, write_ct = 0.0, ct + new_ct
            if read_tool > cr:
                capped += w
                read_tool = cr
                read_ct = min(read_ct, cr)
            write_tool = min(write_tool, cw + inp)
            write_ct = min(write_ct, cw + inp)
            tool_cost = read_tool * p["read"] + write_tool * p["write"]
            ct_cost = read_ct * p["read"] + write_ct * p["write"]
            agg["tool_token_cost"] += w * tool_cost
            for k in ratios:
                oracle_saving[k] += w * (1 - 1 / k) * tool_cost
            realistic["main_saving_r4"] += w * 0.75 * ct_cost
            for x in r["new_tool"]:
                n = x["chars"] * tok_per_char
                if w and n >= INGEST_THRESHOLD:
                    realistic["compressed_results"] += 1
                    realistic["compressed_tokens"] += n
                    realistic["abstractive_cost"] += (n + COMPRESSOR_PROMPT_TOKENS) * HAIKU_IN / 1e6 + (n / 4) * HAIKU_OUT / 1e6
                    realistic["extractive_cost"] += (n + COMPRESSOR_PROMPT_TOKENS) * HAIKU_IN / 1e6 + (0.05 * n) * HAIKU_OUT / 1e6
                    realistic["abstractive_output_tokens"] += n / 4
            tt += new_tool
            ct += new_ct
            prev = r

    total = sum(cost.values())
    return {
        "requests": agg["requests"], "token_totals": {k: agg[k] for k in (
            "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")},
        "cost_usd": dict(cost), "cost_total_usd": total,
        "cost_share": {k: v / total for k, v in cost.items()} if total else {},
        "break_counts": dict(break_counts), "break_write_tokens": dict(break_tokens),
        "break_write_cost_share_of_total": {k: v / total for k, v in break_cost.items()} if total else {},
        "compaction_calls_est_cost_share": cost_compaction_call_est[0] / total if total else 0,
        "break_write_share_excl_hits": {
            k: v / max(1, sum(x for c, x in break_tokens.items() if c != "hit"))
            for k, v in break_tokens.items() if k != "hit"},
        "appended_tokens": agg["appended_tokens"], "appended_tool_tokens": agg["appended_tool_tokens"],
        "tool_share_of_appended": agg["appended_tool_tokens"] / max(1, agg["appended_tokens"]),
        "tool_tokens_by_name": dict(tool_tokens_by_name.most_common(12)),
        "tool_token_cost_usd": agg["tool_token_cost"],
        "tool_token_cost_share": agg["tool_token_cost"] / total if total else 0,
        "oracle_saving_share": {str(k): v / total for k, v in oracle_saving.items()} if total else {},
        "realistic_r4": {
            **{k: v for k, v in realistic.items()},
            "main_saving_share": realistic["main_saving_r4"] / total if total else 0,
            "net_abstractive_share": (realistic["main_saving_r4"] - realistic["abstractive_cost"]) / total if total else 0,
            "net_extractive_share": (realistic["main_saving_r4"] - realistic["extractive_cost"]) / total if total else 0,
        },
        "compactions": compactions, "capped_requests": capped,
        "tokens_per_char_deciles": statistics.quantiles(tpc_samples, n=10) if len(tpc_samples) > 10 else tpc_samples,
        "gap_minutes_quantiles": statistics.quantiles(gaps_min, n=20) if len(gaps_min) > 20 else gaps_min,
        "gaps_over_5m": sum(g > 5 for g in gaps_min), "gaps_over_60m": sum(g > 60 for g in gaps_min),
    }


def verbatim(streams):
    """P4: is each Edit old_string present verbatim in an earlier tool result?"""
    found = collections.Counter()
    lags, tool_lags, samples = [], [], []
    for reqs in streams:
        history = []  # (request index, tool name, normalised text, raw text, compaction epoch)
        epoch = 0
        for i, r in enumerate(reqs):
            if r["compactions"]:
                epoch += 1
            for x in r["new_tool"]:
                norm = READ_PREFIX.sub("", x["text"]) if x["name"] == "Read" else x["text"]
                history.append((i, x["name"], norm, x["text"], epoch))
            if r["dup"]:
                continue
            for old in r["edits"]:
                found["edits"] += 1
                j = next((k for k in range(len(history) - 1, -1, -1) if old in history[k][2]), None)
                src = history[j] if j is not None else None
                if src is None:
                    found["not_found"] += 1
                    continue
                found["found"] += 1
                found["src_" + src[1]] += 1
                lag = i - src[0]
                lags.append(lag)
                tool_lags.append(len(history) - 1 - j)
                if src[4] != epoch:
                    found["across_compaction"] += 1
                if src[1] == "Read" and len(src[3]) <= 8000:
                    samples.append({"read_result": src[3], "old_string": old, "lag": lag})
    q = statistics.quantiles(lags, n=10) if len(lags) > 10 else lags
    return {
        "counts": dict(found),
        "found_share": found["found"] / max(1, found["edits"]),
        "lag_requests_deciles": q,
        "lag_le_1": sum(l <= 1 for l in lags) / max(1, len(lags)),
        "lag_le_3": sum(l <= 3 for l in lags) / max(1, len(lags)),
        "lag_le_10": sum(l <= 10 for l in lags) / max(1, len(lags)),
        "tool_lag_deciles": statistics.quantiles(tool_lags, n=10) if len(tool_lags) > 10 else tool_lags,
        "tool_lag_lt_3": sum(l < 3 for l in tool_lags) / max(1, len(tool_lags)),
        "tool_lag_lt_5": sum(l < 5 for l in tool_lags) / max(1, len(tool_lags)),
    }, samples


def verbatim_with_self_sources(streams):
    """Post hoc (added after E-24's first run, not predeclared): the most recent verbatim
    source of each Edit old_string, counting the agent's own earlier Write and Edit text as a
    competing source. Only edits whose latest source is a tool result depend on that result."""
    c = collections.Counter()
    tool_lags = []
    for reqs in streams:
        history = []  # (kind, normalised text)
        for r in reqs:
            for x in r["new_tool"]:
                norm = READ_PREFIX.sub("", x["text"]) if x["name"] == "Read" else x["text"]
                history.append(("tool", norm))
            if not r["dup"]:
                for old in r["edits"]:
                    c["edits"] += 1
                    j = next((k for k in range(len(history) - 1, -1, -1) if old in history[k][1]), None)
                    if j is None:
                        c["no_source"] += 1
                    elif history[j][0] == "self":
                        c["latest_source_self"] += 1
                    else:
                        c["latest_source_tool"] += 1
                        tool_lags.append(sum(1 for h in history[j + 1:] if h[0] == "tool"))
            history.extend(("self", t) for t in r["self_texts"] if t)
    n = max(1, c["edits"])
    return {
        "counts": dict(c),
        "tool_source_share": c["latest_source_tool"] / n,
        "self_source_share": c["latest_source_self"] / n,
        "tool_lag_deciles": statistics.quantiles(tool_lags, n=10) if len(tool_lags) > 10 else tool_lags,
        "tool_lag_lt_3": sum(l < 3 for l in tool_lags) / max(1, len(tool_lags)),
        "tool_lag_lt_5": sum(l < 5 for l in tool_lags) / max(1, len(tool_lags)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-out", required=True)
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.root, "**", "*.jsonl"), recursive=True)
             if EXCLUDE_SESSION not in f]
    # Forked or resumed sessions copy earlier requests into a new file. A request already
    # seen in an earlier file still updates per-stream state but is not counted again.
    main_streams, side_streams, counted = [], [], set()
    for f in sorted(files):
        for sidechain, bucket in ((False, main_streams), (True, side_streams)):
            stream = load_stream(f, sidechain=sidechain)
            for r in stream:
                r["dup"] = r["rid"] in counted
                counted.add(r["rid"])
            if stream:
                bucket.append(stream)

    prompt_sizes = [max(sum(r["usage"].get(k, 0) or 0 for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")) for r in s)
        for s in main_streams]
    res = {
        "files": len(files), "main_sessions": len(main_streams), "sidechain_streams": len(side_streams),
        "session_max_prompt_quantiles": statistics.quantiles(prompt_sizes, n=10),
        "sessions_max_prompt_over": {str(t): sum(p > t for p in prompt_sizes) for t in (150_000, 200_000, 500_000, 900_000)},
        "main": analyse(main_streams),
        "sidechain": analyse(side_streams),
    }
    v, samples = verbatim(main_streams)
    res["verbatim"] = v
    res["verbatim"]["read_samples_le_8000_chars"] = len(samples)
    res["verbatim_posthoc_self_sources"] = verbatim_with_self_sources(main_streams)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    random.Random(0).shuffle(samples)
    with open(args.samples_out, "w") as fh:
        json.dump(samples, fh)
    print(json.dumps(res, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
