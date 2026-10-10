"""Extract per-call timing of agent sessions from the Exgentic agent-llm-traces shards.

Output: one CSV row for each LLM call (span) with the input and output token counts,
the call duration, and the gap since the previous call of the same session ended
(the tool and think time). The simulator for E-28 samples sessions from this file.

Run with the trace venv (it has pyarrow):
  ~/Downloads/agent-llm-traces/.venv/bin/python -I extract_sessions.py <data_dir> <out_csv>
"""
import csv
import glob
import os
import sys
from datetime import datetime

import pyarrow.parquet as pq

COLS = [
    "harness", "benchmark", "models.list.element", "session_id",
    "spans.list.element.name", "spans.list.element.start_time", "spans.list.element.end_time",
    "spans.list.element.attributes.gen_ai.operation.name",
    "spans.list.element.attributes.gen_ai.usage.input_tokens",
    "spans.list.element.attributes.gen_ai.usage.output_tokens",
]


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def main(data_dir, out_csv):
    n_sess = n_calls = n_skip = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "harness", "benchmark", "model", "call_idx", "start_s",
                    "dur_s", "gap_before_s", "overlaps_prev", "in_tok", "out_tok"])
        for path in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
            for row in pq.ParquetFile(path).read(columns=COLS).to_pylist():
                calls = []
                for sp in row["spans"] or []:
                    a = sp["attributes"] or {}
                    if a.get("gen_ai.operation.name") != "chat":
                        continue
                    if a.get("gen_ai.usage.input_tokens") is None or not sp["start_time"]:
                        n_skip += 1
                        continue
                    calls.append((ts(sp["start_time"]), ts(sp["end_time"]),
                                  a["gen_ai.usage.input_tokens"],
                                  a.get("gen_ai.usage.output_tokens") or 0))
                if not calls:
                    continue
                calls.sort()
                t0 = calls[0][0]
                prev_end = None
                model = (row["models"] or [""])[0]
                for i, (s, e, it, ot) in enumerate(calls):
                    # A negative gap means the call started before the previous one ended
                    # (parallel subagents). Keep the row and flag it.
                    gap = "" if prev_end is None else round(s - prev_end, 3)
                    ovl = int(prev_end is not None and s < prev_end)
                    w.writerow([row["session_id"], row["harness"], row["benchmark"], model, i,
                                round(s - t0, 3), round(e - s, 3), gap, ovl, it, ot])
                    prev_end = e if prev_end is None else max(prev_end, e)
                n_sess += 1
                n_calls += len(calls)
    print(f"sessions={n_sess} calls={n_calls} skipped_spans={n_skip}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
