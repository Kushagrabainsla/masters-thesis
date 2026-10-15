"""Build the references/ folder from RESEARCH_JOURNAL.md Section 7 and the dossiers.

Run: python3 references/_tools/build_references.py
The script finds every arXiv ID in the journal, the dossiers and the experiment READMEs.
It gets titles and authors from the arXiv API and caches them in arxiv_metadata.json.
If the cache has every ID, the script does not use the network.
The journal is the source of truth. Do not edit the generated files by hand.
"""
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
OUT = TOOLS.parent
REPO = OUT.parent
CACHE = TOOLS / "arxiv_metadata.json"

# Journal Section 7 subsection number -> folder name.
TOPICS = {
    "7.1": "01-numerics-kernel-dispatch",
    "7.2": "02-hybrid-state-prefix-caching",
    "7.3": "03-collectives-communication",
    "7.4": "04-migration-checkpoint-fault-tolerance",
    "7.5": "05-rl-rollout-mismatch",
    "7.6": "06-elastic-training-traces",
    "7.7": "07-handoff-frontier-signals",
    "7.8": "08-agentic-harnesses",
    "7.9": "09-gpu-efficiency",
    "7.10": "10-agent-context-compression",
    "7.11": "11-cross-domain-directions-r09",
    "7.12": "12-mechanism-transfer-routes-r10-r13",
    "7.13": "13-route-search-r17",
    "7.14": "14-orphan-scan-r18",
    "R-19": "15-assumption-removal-r19",
    "other": "16-other-journal-citations",
}
EXTRA_TITLES = {
    "R-19": "Assumption removal (R-19, 2026-10-11). Not yet in journal Section 7",
    "other": "Other papers cited in the journal outside Section 7",
}

# If a paper is not in the ledger, the first dossier that cites it picks its topic.
DOSSIER_TOPIC = [
    ("LIT_REVIEW_GPU_EFFICIENCY", "7.9"),
    ("LIT_REVIEW_AGENTIC_HARNESSES", "7.8"),
    ("DOSSIER_I-41", "7.10"),
    ("NEW_DIRECTIONS_R-09", "7.11"),
    ("ROUTES_R-10", "7.12"),
    ("REVIEW_R-14", "7.12"),
    ("REVIEW_E-28", "7.12"),
    ("REVIEW_R-16", "7.12"),
    ("R-17", "7.13"),
    ("REVIEW_R-17", "7.13"),
    ("ROUTES_R-17", "7.13"),
    ("ORPHANS_R-18", "7.14"),
    ("R-18_", "7.14"),
    ("REVIEW_R-18", "7.14"),
    ("R-19", "R-19"),
    ("ASSUMPTIONS_R-19", "R-19"),
    ("REVIEW_R-19", "R-19"),
]

ID_RE = re.compile(r"(?<![\d.])(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?(?!\d)")


def ids_in(text):
    out = []
    for m in ID_RE.finditer(text):
        yy, mm, num = int(m.group(1)), int(m.group(2)), m.group(3)
        if not 1 <= mm <= 12:
            continue
        if len(num) == 5 and yy < 15:
            continue
        if len(num) == 4 and not 7 <= yy <= 14:
            continue
        aid = f"{m.group(1)}{m.group(2)}.{num}"
        if aid not in out:
            out.append(aid)
    return out


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_ledger(journal):
    sec7 = journal[journal.index("## 7. Prior-art ledger"):journal.index("## 8. Errors")]
    ledger, prose, titles = {}, {}, {}
    cur, header = None, None
    for line in sec7.splitlines():
        if line.startswith("### 7."):
            num, title = line[4:].split(" ", 1)
            cur, header = num, None
            ledger[cur], prose[cur], titles[cur] = [], [], title
            continue
        if cur is None:
            continue
        if line.startswith("|"):
            cells = split_row(line)
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if header is None:
                header = cells
                continue
            ledger[cur].append({"row": dict(zip(header, cells)), "ids": ids_in(line)})
        elif line.strip() and not line.startswith("Cumulative"):
            prose[cur].append(line.strip())
    return ledger, prose, titles


def collect_citations():
    """Return {id: {relpath: (section, context line)}}."""
    cites = {}
    files = [REPO / "RESEARCH_JOURNAL.md"] + sorted((REPO / "dossier").glob("*.md")) + sorted(
        (REPO / "experiments").glob("*/README.md"))
    for f in files:
        rel = str(f.relative_to(REPO))
        section = ""
        for line in f.read_text().splitlines():
            if line.startswith("## ") or line.startswith("### "):
                section = line.lstrip("#").strip()
            for aid in ids_in(line):
                cites.setdefault(aid, {}).setdefault(rel, (section, line.strip()))
    return cites


def fetch_metadata(ids):
    meta = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [i for i in sorted(ids) if not meta.get(i)]
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        url = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(batch)
        data = urllib.request.urlopen(url, timeout=60).read()
        for e in ET.fromstring(data).findall("a:entry", ns):
            aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1].split("v")[0]
            title = " ".join(e.findtext("a:title", "", ns).split())
            if aid in batch and title:
                cat = e.find("x:primary_category", ns)
                meta[aid] = {
                    "title": title,
                    "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                    "published": e.findtext("a:published", "", ns)[:10],
                    "category": cat.get("term") if cat is not None else "",
                }
        time.sleep(3.5)  # arXiv API rate limit
    CACHE.write_text(json.dumps(dict(sorted(meta.items())), indent=1) + "\n")
    return meta


def cell(text, limit=None):
    text = re.sub(r"\s+", " ", text.replace("|", "/")).strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


def authors(a):
    return ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")


def cited_in(files):
    links = []
    for rel, (section, _) in files.items():
        name = Path(rel).stem if rel != "RESEARCH_JOURNAL.md" else "Journal"
        if rel == "RESEARCH_JOURNAL.md" and section:
            name += " " + section.split(" ", 1)[0].rstrip(".")
        links.append(f"[{name}](../../{rel})")
    return ", ".join(links)


def paper_cell(aid, meta):
    m = meta[aid]
    return f"[{cell(m['title'])}](https://arxiv.org/abs/{aid}) `{aid}`"


def main():
    journal = (REPO / "RESEARCH_JOURNAL.md").read_text()
    ledger, prose, titles = parse_ledger(journal)
    cites = collect_citations()
    meta = fetch_metadata(cites)

    # Topic of each paper: the first ledger section that lists it, else the first citing dossier.
    topic_of, notes = {}, {}
    for sec, rows in ledger.items():
        for r in rows:
            for aid in r["ids"]:
                topic_of.setdefault(aid, sec)
                row = r["row"]
                used = "; ".join(x for x in (row.get("What it shows", ""), row.get("Effect", "")) if x)
                notes.setdefault(aid, (used or row.get("Title", ""), row.get("Check", "")))
    for sec, lines in prose.items():
        level = "doc-only" if "doc-only" in titles[sec] else "search-only"
        for aid in ids_in(" ".join(lines)):
            topic_of.setdefault(aid, sec)
            notes.setdefault(aid, ("Named in the ledger prose for this topic (see below)", level))
    for aid, files in cites.items():
        if aid in topic_of:
            continue
        for rel in files:
            stem = Path(rel).stem
            hit = next((t for prefix, t in DOSSIER_TOPIC if stem.startswith(prefix)), None)
            if hit:
                topic_of[aid] = hit
                break
        else:
            topic_of[aid] = "other"

    index_rows = []
    for key, folder in TOPICS.items():
        title = titles.get(key) or EXTRA_TITLES[key]
        papers = sorted((a for a, t in topic_of.items() if t == key), key=lambda a: meta[a]["published"])
        other_rows = [r for r in ledger.get(key, []) if not r["ids"]]
        lines = [f"# {title}", ""]
        if key.startswith("7."):
            lines.append(f"Source: `RESEARCH_JOURNAL.md` Section {key}. Papers that only a dossier cites are added below the ledger papers.")
        elif key == "R-19":
            lines.append("Source: the R-19 files in `dossier/`. These papers are not yet in journal Section 7.")
        else:
            lines.append("Source: `RESEARCH_JOURNAL.md` outside Section 7 (idea dossiers, experiments, reviews).")
        lines += ["", "Check levels are defined in journal Section 0.4. \"dossier\" or \"journal\" means the paper is not in the ledger, so it has no check level.", ""]
        if papers:
            lines += [f"## Papers ({len(papers)})", "",
                      "| Paper | Authors | Date | What we used it for | Check | Cited in |",
                      "|---|---|---|---|---|---|"]
            for aid in papers:
                m = meta[aid]
                if aid in notes:
                    used, check = notes[aid]
                else:
                    files = cites.get(aid, {})
                    rel, (_, line) = next(iter(files.items()), ("", ("", "")))
                    if line.startswith("|"):
                        line = "; ".join(c for c in split_row(line) if c)
                    used, check = "Note in the citing file: " + line.lstrip("-* "), (
                        "journal" if rel == "RESEARCH_JOURNAL.md" else "dossier")
                lines.append(f"| {paper_cell(aid, meta)} | {cell(authors(m['authors']))} | {m['published']} | "
                             f"{cell(used, 300)} | {cell(check)} | {cited_in(cites.get(aid, {}))} |")
            lines.append("")
        if other_rows:
            lines += [f"## Other sources without an arXiv ID ({len(other_rows)})", "",
                      "GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.", "",
                      "| Work | What it shows | Check |", "|---|---|---|"]
            for r in other_rows:
                row = r["row"]
                work = row.get("Work") or row.get("Title") or ""
                used = "; ".join(x for x in (row.get("What it shows", ""), row.get("Effect", "")) if x)
                lines.append(f"| {cell(work)} | {cell(used, 300)} | {cell(row.get('Check', ''))} |")
            lines.append("")
        if prose.get(key) and not ledger.get(key):
            lines += ["## Other works named in the ledger", ""] + [" ".join(prose[key]), ""]
        (OUT / folder).mkdir(parents=True, exist_ok=True)
        (OUT / folder / "README.md").write_text("\n".join(lines))
        index_rows.append((folder, title, len(papers), len(other_rows)))

    bib = []
    for aid in sorted(meta):
        m = meta[aid]
        auth = " and ".join(m["authors"]).replace("{", "").replace("}", "")
        bib.append(f"@misc{{arxiv{aid.replace('.', '_')},\n  title = {{{{{m['title']}}}}},\n  author = {{{auth}}},\n"
                   f"  year = {{{m['published'][:4]}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n"
                   f"  primaryClass = {{{m['category']}}},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}\n")
    (OUT / "references.bib").write_text("\n".join(bib))

    readme = (OUT / "README.md").read_text()
    table = ["| Folder | Topic | arXiv papers | Other sources |", "|---|---|---|---|"]
    table += [f"| [`{f}/`]({f}/README.md) | {cell(t)} | {n} | {o} |" for f, t, n, o in index_rows]
    total = sum(r[2] for r in index_rows)
    table.append(f"| | **Total** | **{total}** | **{sum(r[3] for r in index_rows)}** |")
    start, end = "<!-- index:start -->", "<!-- index:end -->"
    readme = readme[:readme.index(start) + len(start)] + "\n" + "\n".join(table) + "\n" + readme[readme.index(end):]
    (OUT / "README.md").write_text(readme)
    print(f"{total} arXiv papers in {len(index_rows)} folders")


if __name__ == "__main__":
    main()
