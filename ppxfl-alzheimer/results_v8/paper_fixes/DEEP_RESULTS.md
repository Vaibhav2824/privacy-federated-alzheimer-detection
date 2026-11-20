# Fix 2: deep-learning baseline: NOT RUN (stopped at Step 1)

Step 1 requires a registered T1 volume for every one of the 1,262 subjects in
`data/v8/features_v8_dp27.npz`. On this machine (Apple M5 Pro, 24 GB, no CUDA;
PyTorch MPS only):

- 474 of 1,262 subjects have a volume here (`data/mni2mm_v8`, `data/mni2mm_v8b`),
  and those are skull-stripped / oriented scans, not the MNI-registered volumes
  the slice builder expects.
- 788 subjects have no volume here: mostly the original v7 cohort, whose
  registered volumes were produced on another machine (`data/mni2mm` here holds
  only the feature files and manifest).

Per the rules in COAUTHOR_FIXES.md ("If any subject is missing a volume, STOP"),
nothing was trained. The missing-subject list is not written to git (it is
participant-level); it can be regenerated locally by comparing `subjects` in
`features_v8_dp27.npz` with the volume folders.

To run Fix 2: use the machine that holds the MNI-registered volumes for all
1,262 subjects and a CUDA GPU, and follow Steps 1-4 of COAUTHOR_FIXES.md unchanged.
