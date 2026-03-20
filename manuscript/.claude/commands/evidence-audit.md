Audit the empirical claims in the manuscript against available evidence. Context: $ARGUMENTS

## Steps

1. Read the manuscript and identify all empirical/computational claims.
2. Read any available results files (CSVs, tables, logs) in the project.
3. Build a claim-evidence table mapping each claim to its support.
4. Check statistical rigor for each claim-facing comparison.

## Claim-Evidence Table

For each claim, record:
| Claim ID | Claim Text | Metric | Dataset | Reported Value | Evidence Source | Status | Action Needed |
|---|---|---|---|---|---|---|---|

Status values: `supported`, `outside_tolerance`, `no_match`, `manual_review`

## Statistical Rigor Check

For every claim-facing result, check:
- Is uncertainty/confidence interval reported?
- Are multi-seed runs used for stochastic methods?
- Are paired comparisons used when same seeds/splits are available?
- Is metric direction explicit?
- Are effect sizes reported alongside p-values?

## When to Flag

Flag and recommend weakening claim language when:
- The interval crosses zero or the practical equivalence threshold
- Gains appear only on one dataset or one seed pattern
- One stronger baseline removes the claimed advantage
- The result depends on unusually favorable tuning budget
- Only point estimates are reported without uncertainty

## Output

1. Claim-evidence table
2. Statistical rigor assessment per claim
3. Unsupported or weak claims that need attention
4. Recommended claim language changes
5. Missing experiments or analyses needed
