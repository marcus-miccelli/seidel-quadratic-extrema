# Contributing

This repository is intended to remain a reliable research handoff rather than an accumulation of speculative notes.

Every mathematical change must identify its status as one of:

- **proved** — a complete argument is included or precisely cited;
- **exhaustive computation** — completeness, arithmetic, software revision, and independent verification are documented;
- **numerical observation** — evidence only, with no theorem claim;
- **open** — a question or explicitly incomplete argument;
- **refuted** — retained because it closes a plausible route.

For a new theorem, include its normalization ledger and check every conversion between the pair-sum objective $Q_A$, the full quadratic form $x^{\mathsf T}Ax$, and $M(A)$. For computational results, provide a deterministic command, compact output, input hashes or generation rules, and an independent check that does not repeat the same implementation.

Please keep generated binaries, caches, raw search logs, and large intermediate enumeration levels out of Git. Open a focused issue before undertaking a large computation so that its certificate format and stopping rule can be agreed in advance.

Before proposing a change, run:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
latexmk -pdf -cd paper/report.tex
```
