Perform a full draft diagnosis on the manuscript. The target is: $ARGUMENTS

## Steps

1. Read the manuscript file (PDF, LaTeX, markdown, or text as available in this directory).
2. Build a working record: identify venue, stage, paper type, datasets, metrics, and headline claims.
3. Diagnose before rewriting — separate scientific risk, reviewer risk, compliance risk, and polish risk.

## Produce this output

```
# Paper Diagnosis

## Verdict
[One paragraph on current readiness, strongest asset, and biggest acceptance risk.]

## Blocking Issues
1. [Issue] - [why it matters] - [minimum fix]

## Important Issues
1. ...

## Polish
1. ...

## Claim-Evidence Table
| Claim | Current Support | Risk | Action |
|---|---|---|---|

## Revision Plan
1. [section or artifact] -> [specific change] -> [expected reviewer impact]

## Experiment and Statistics Plan
- [new baseline, ablation, robustness run, or uncertainty analysis]
- [stop condition]
- [expected decision impact]

## Submission Readiness
- [required statements, artifacts, and venue checks still missing]
```

## Guidelines
- Score through reviewer lenses: soundness, clarity, significance, originality, reproducibility, responsible research.
- For each dimension: 1-2 sentences, name highest-risk weakness, propose smallest change that materially improves score.
- Distinguish what can be fixed by writing from what requires new evidence.
- Use red-flag triage order from CLAUDE.md.
