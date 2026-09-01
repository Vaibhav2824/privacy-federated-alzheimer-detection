"""Structural sanity checks on paper.tex: environments, labels, refs, citations.

These are the failures that a reader does not see and a compile reports far
from their cause. A row terminated with one backslash instead of two, for
instance, surfaces as a misplaced \\noalign several lines later, at whichever
rule happens to follow it.
"""

import collections
import re
import sys

PATH = "../paper.tex"


def undefined_macros(text: str) -> list:
    """Macros invoked as \\Name{} that nothing defines.

    The paper's numbers live in \\newcommand definitions generated from the
    results, so a macro that lost its definition prints nothing and silently
    removes a figure from a sentence.
    """
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))
    body = re.sub(r"\\newcommand\{\\[A-Za-z]+\}\{[^}]*\}", "", text)
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", body))
    # \LaTeX and friends come from the class, not from this document.
    builtin = {"LaTeX", "TeX", "LaTeXe", "today", "hfill", "noindent"}
    return sorted(used - defined - builtin)


def stale_macros(text: str) -> list:
    """Definitions no reader reaches.

    An orphaned definition is not merely clutter: it holds a number that the
    prose beside it has already moved past, and the next person to grep the
    source finds both.
    """
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))
    body = re.sub(r"\\newcommand\{\\[A-Za-z]+\}\{[^}]*\}", "", text)
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", body))
    return sorted(defined - used)


def unterminated_rows(text: str) -> list:
    """Table rows ending in an odd number of backslashes."""
    bad = []
    for number, line in enumerate(text.split("\n"), 1):
        if "&" not in line or not line.endswith("\\"):
            continue
        trailing = len(line) - len(line.rstrip("\\"))
        if trailing % 2 == 1:
            bad.append(f"{number}: {line.strip()[:60]}")
    return bad


def main() -> int:
    text = open(PATH, encoding="utf-8").read()

    begins = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", text))
    ends = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
    unbalanced = {k: begins[k] - ends[k] for k in set(begins) | set(ends)
                  if begins[k] != ends[k]}

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    duplicates = [k for k, v in collections.Counter(labels).items() if v > 1]
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", text))
    dangling = sorted(refs - set(labels))

    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    cites = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", text):
        cites |= {c.strip() for c in group.split(",")}
    uncited = sorted(cites - bibitems) if bibitems else []

    markers = re.findall(r"% (BEGIN|END) AUTO:([\w-]+)", text)
    starts = [m[1] for m in markers if m[0] == "BEGIN"]
    stops = [m[1] for m in markers if m[0] == "END"]

    rows = unterminated_rows(text)
    missing = undefined_macros(text)
    stale = stale_macros(text)

    print("unbalanced environments:", unbalanced or "none")
    print("duplicate labels:", duplicates or "none")
    print("refs with no label:", dangling or "none")
    print("cites with no bibitem:", uncited or "none")
    print("unterminated table rows:", rows or "none")
    print("macros used but not defined:", missing or "none")
    print("macros defined but unused:", stale or "none")
    print(f"AUTO markers: {len(starts)} begin / {len(stops)} end",
          "(matched)" if sorted(starts) == sorted(stops) else "(MISMATCH)")

    ok = (not unbalanced and not duplicates and not dangling and not uncited
          and not rows and not missing and not stale
          and sorted(starts) == sorted(stops))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
