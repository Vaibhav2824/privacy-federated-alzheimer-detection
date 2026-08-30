"""Structural sanity checks on paper.tex: environments, labels, refs, citations."""

import collections
import re
import sys

PATH = "../paper.tex"


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

    print("unbalanced environments:", unbalanced or "none")
    print("duplicate labels:", duplicates or "none")
    print("refs with no label:", dangling or "none")
    print("cites with no bibitem:", uncited or "none")
    print(f"AUTO markers: {len(starts)} begin / {len(stops)} end",
          "(matched)" if sorted(starts) == sorted(stops) else "(MISMATCH)")

    ok = not unbalanced and not duplicates and not dangling and not uncited \
        and sorted(starts) == sorted(stops)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
