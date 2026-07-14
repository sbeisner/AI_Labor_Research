#!/usr/bin/env python
"""Pre-render regression test for `tas_manuscript.qmd`.

Four gates, run in order; the script exits non-zero if any gate fails:

  1. EXECUTION   — extract every ```{python} block and execute it headlessly
                   (matplotlib Agg). Any exception is a failure.
  2. CROSS-REFS  — every @fig-/@tbl-/@eq-/@sec- reference (and every LaTeX
                   \\autoref{...}) must resolve to a defined label.
  3. CITATIONS   — every @citekey in the prose must exist in bibliography.bib.
  4. STATS       — regenerate output/stats.json via scripts/extract_stats.py and
                   fail if any quoted prose statistic has drifted from the value
                   recomputed from the current output files (prose–data lockstep).

Run before every render:  python scripts/check_manuscript.py
Or via the Makefile:       make check-manuscript
"""

import os
import re
import sys
import pathlib
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
QMD = ROOT / "tas_manuscript.qmd"
SUPP = ROOT / "supplement.qmd"
BIB = ROOT / "bibliography.bib"

CROSSREF_PREFIXES = ("fig-", "tbl-", "eq-", "sec-")


def _defined_labels(text):
    """All cross-reference labels a document defines (div/heading, cell, \\label)."""
    d = set(re.findall(r"\{#((?:sec|fig|tbl|eq)-[A-Za-z0-9_-]+)", text))
    d |= set(re.findall(r"^#\|\s*label:\s*((?:sec|fig|tbl|eq)-[A-Za-z0-9_-]+)", text, re.M))
    d |= set(re.findall(r"\\label\{([A-Za-z0-9_:-]+)\}", text))
    return d


# ── Block extraction ──────────────────────────────────────────────────────────
def extract_blocks(text, lang):
    """Return list of (start_line, code) for every ```{lang} ... ``` fenced block."""
    blocks = []
    lines = text.splitlines()
    i = 0
    open_re = re.compile(r"^```\{" + re.escape(lang) + r"\}\s*$")
    while i < len(lines):
        if open_re.match(lines[i]):
            start = i + 1
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append((start, "\n".join(body)))
        i += 1
    return blocks


# ── Gate 1: execute every python block ────────────────────────────────────────
def gate_execution(text, label="tas_manuscript.qmd"):
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # plt.show() is a harmless no-op under Agg; silence its warning.
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

    blocks = extract_blocks(text, "python")
    print(f"[exec:{label}] Executing {len(blocks)} python block(s) headlessly (Agg) ...")
    failures = []
    cwd = os.getcwd()
    os.chdir(ROOT)  # relative paths like output/*.parquet resolve from repo root
    try:
        for n, (start_line, code) in enumerate(blocks, 1):
            label_m = re.search(r"^#\|\s*label:\s*(\S+)", code, re.M)
            label = label_m.group(1) if label_m else f"block#{n}"
            ns = {"__name__": "__manuscript_block__"}
            try:
                exec(compile(code, f"{label}:{start_line}", "exec"), ns)
                plt.close("all")
                print(f"      ok   {label}")
            except Exception:  # noqa: BLE001
                plt.close("all")
                failures.append((label, start_line, traceback.format_exc()))
                print(f"      FAIL {label}  (qmd line {start_line})")
    finally:
        os.chdir(cwd)

    for label, start_line, tb in failures:
        print(f"\n--- {label} (qmd line {start_line}) ---\n{tb}")
    return not failures


# ── Gate 2: cross-reference resolution ────────────────────────────────────────
def gate_crossrefs(text, label="tas_manuscript.qmd", also_defined=frozenset()):
    print(f"[refs:{label}] Checking cross-reference resolution ...")

    # A ref resolves if defined in this doc OR its companion (the two are a
    # linked pair: the supplement points at main-text sections and vice versa).
    defined = _defined_labels(text) | set(also_defined)

    # References: @fig-/@tbl-/@eq-/@sec- and \autoref{...}
    referenced = {}
    for m in re.finditer(r"(?<![A-Za-z0-9])@((?:sec|fig|tbl|eq)-[A-Za-z0-9_-]+)", text):
        referenced.setdefault(m.group(1), text[:m.start()].count("\n") + 1)
    for m in re.finditer(r"\\autoref\{([A-Za-z0-9_:-]+)\}", text):
        referenced.setdefault(m.group(1), text[:m.start()].count("\n") + 1)

    unresolved = {r: ln for r, ln in referenced.items() if r not in defined}
    print(f"      {len(defined)} labels defined, {len(referenced)} references seen")
    for r, ln in sorted(unresolved.items(), key=lambda kv: kv[1]):
        print(f"      UNRESOLVED  @{r}  (first seen qmd line {ln})")
    return not unresolved


# ── Gate 3: citation existence ────────────────────────────────────────────────
def gate_citations(text, label="tas_manuscript.qmd"):
    print(f"[cite:{label}] Checking citation keys against bibliography.bib ...")
    bib = BIB.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"^@\w+\{\s*([^,\s]+)\s*,", bib, re.M))

    referenced = {}
    # A citation key: @key not preceded by an alphanumeric (excludes emails),
    # and not one of the cross-reference prefixes.
    for m in re.finditer(r"(?<![A-Za-z0-9@/])@([A-Za-z][A-Za-z0-9_:.-]+)", text):
        key = m.group(1).rstrip(".:-")
        if key.startswith(CROSSREF_PREFIXES):
            continue
        referenced.setdefault(key, text[:m.start()].count("\n") + 1)

    missing = {k: ln for k, ln in referenced.items() if k not in bib_keys}
    print(f"      {len(bib_keys)} bib keys, {len(referenced)} distinct citations cited")
    for k, ln in sorted(missing.items(), key=lambda kv: kv[1]):
        print(f"      MISSING  @{k}  (first cited qmd line {ln})")
    return not missing


# ── Gate 4: prose–data lockstep (stats.json drift) ────────────────────────────
def gate_stats():
    """Fail if any quoted prose statistic has drifted from the value recomputed
    from the current output files. Regenerates output/stats.json first."""
    import json
    import subprocess

    print("[stats] Checking prose–data lockstep (output/stats.json) ...")
    stats_json = ROOT / "output" / "stats.json"
    extract = ROOT / "scripts" / "extract_stats.py"
    # Regenerate so the check reflects the current output files, not a cached JSON.
    try:
        subprocess.run([sys.executable, str(extract)], check=True,
                       capture_output=True, text=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        print(f"      could not run extract_stats.py:\n{e.stderr}")
        return False
    if not stats_json.exists():
        print("      output/stats.json not found after extract_stats.py")
        return False

    stats = json.loads(stats_json.read_text())
    drift = {k: v for k, v in stats.items()
             if abs(v["value"] - v["quoted"]) > v["tol"]}
    print(f"      {len(stats)} tracked statistics")
    for k, v in sorted(drift.items(), key=lambda kv: kv[1]["where"]):
        print(f"      DRIFT  {k} [{v['where']}]: computed {v['value']} vs "
              f"quoted {v['quoted']} (tol {v['tol']})")
    return not drift


def main():
    if not QMD.exists():
        sys.exit(f"error: {QMD} not found")
    if not SUPP.exists():
        sys.exit(f"error: {SUPP} not found")
    main_text = QMD.read_text(encoding="utf-8")
    supp_text = SUPP.read_text(encoding="utf-8")

    main_labels = _defined_labels(main_text)
    supp_labels = _defined_labels(supp_text)

    results = {}
    # Main manuscript: all four gates.
    results["exec:main"] = gate_execution(main_text, QMD.name)
    results["refs:main"] = gate_crossrefs(main_text, QMD.name, supp_labels)
    results["cite:main"] = gate_citations(main_text, QMD.name)
    results["stats"] = gate_stats()
    # Supplement: execution, cross-refs, citations (no separate stats lockstep —
    # its figures compute their own numbers live from the same output files).
    results["exec:supp"] = gate_execution(supp_text, SUPP.name)
    results["refs:supp"] = gate_crossrefs(supp_text, SUPP.name, main_labels)
    results["cite:supp"] = gate_citations(supp_text, SUPP.name)

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:12s}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
