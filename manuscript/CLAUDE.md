# Paper Improvement Instructions

When working on the manuscript in this directory, follow the paper improvement framework below.

## Operating Modes

Pick the mode that matches what's available:

1. **Draft diagnosis** - User has a draft/PDF/LaTeX/text and wants stronger writing, framing, structure, or positioning.
2. **Evidence audit** - Paper makes empirical claims and user has results tables, CSVs, configs, logs, or code.
3. **Review response** - User has reviewer comments or meta-review and needs a response plan plus manuscript changes.
4. **Submission readiness** - User names a venue/deadline and wants a final pass for checklist, formatting, reproducibility, artifacts.

## How to Reason About Paper Quality

Use these priorities in order:

### 1. Claim Support
- Check whether main claims are actually supported by experiments, proofs, analyses, or citations.
- Downgrade novelty or performance language when support is thin.
- Refuse to strengthen a claim if evidence marks it as unsupported or manual-review only.

### 2. Reviewer-Facing Risk
Score the paper through these lenses:

**Soundness**: Are claims explicit? Is methodology appropriate? Are experiments sufficient? Baselines fair and current? Evaluation reproducible?
- Failure modes: claims stronger than evidence, weak baselines, no ablations, no uncertainty/seeds, selective reporting

**Clarity**: Can a reviewer summarize problem/gap/method/result after one read? Does each section have a distinct job?
- Failure modes: generic abstract, mixed intro sections, method without rationale, results without interpretation

**Significance**: Does the paper solve a problem that matters? Do gains matter beyond one benchmark?
- Failure modes: too incremental, no explanation of why result matters, fragile gains

**Originality**: Is closest prior work identified? Is the difference precise rather than rhetorical?
- Failure modes: novelty described with adjectives not contrasts, missing dangerous comparisons

**Reproducibility**: Are datasets/splits/metrics/preprocessing stated? Seeds/uncertainty reported? Code/artifact plan?
- Failure modes: hidden preprocessing, missing code availability, only point estimates

**Responsible Research**: Are limitations specific? Are risks/biases addressed? Failure cases acknowledged?
- Failure modes: generic limitations, no dataset bias discussion, disconnected ethics language

### 3. Narrative Fit
- Make the paper easy to summarize in one paragraph.
- Tighten the problem-gap-method-evidence-takeaway chain.
- Remove generic hype, repeated novelty claims, unsupported adjectives.

### 4. Empirical Rigor
- Do not treat point estimates as enough for claim-facing comparisons.
- Use multi-seed runs when method/training is stochastic.
- Use paired comparisons when same seeds/splits are available.
- Ask for uncertainty, paired analysis, failure cases, ablations, and strong baselines before recommending stronger conclusions.

### 5. Venue Compliance
- For current requirements, deadlines, page limits, anonymity rules, and mandatory checklists, check official venue pages rather than relying on memory.

## Default Outputs

Unless the user asks for something narrower, produce:
1. One-paragraph verdict on current state
2. Prioritized issues split into `blocking`, `important`, and `polish`
3. A revision plan with concrete manuscript changes
4. A claim-evidence table
5. For empirical papers: an experiment and statistics plan
6. For reviewed papers: a point-by-point rebuttal matrix
7. For near-submission papers: a readiness checklist

## Red-Flag Triage Order
1. Unsupported main claim
2. Missing or weak baselines
3. No uncertainty or seed robustness for empirical results
4. Novelty claim not differentiated from closest prior work
5. Missing limitations, ethics, or responsible-use discussion when venue expects it
6. Missing data, code, or artifact availability plan
7. Unclear contribution list or problem statement
8. Polished writing over unstable evidence

## Rewrite Policy
- Tie every major recommendation to a section, claim, figure, table, or reviewer point.
- Prefer scoped claims over broad marketing language.
- Replace vague novelty statements with exact differentiators.
- Never claim significance without statistical or methodological support.
- Rewrite abstract, intro, related work, method framing, and results discussion only after evidence picture is stable.

## Escalation Points
Require user to make a scientific choice when:
- Main claim is unsupported and cannot be fixed with writing alone
- Strongest baseline changes the headline conclusion
- Venue rules conflict with current paper structure
- Additional experiments would materially change claims but exceed budget/deadline

## Reference Templates

Available in `Paper-improvement-Skill/` for templates and checklists. Use `/critique`, `/rewrite`, `/rebuttal`, `/readiness`, or `/evidence-audit` slash commands for specific workflows.
