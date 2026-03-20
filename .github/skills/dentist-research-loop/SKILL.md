---
name: dentist-research-loop
description: Fully autonomous Dentist-specific research loop adapted from ARIS. Uses Copilot custom-agent review via the local aris-reviewer agent on claude-opus-4.6 instead of Codex MCP, resumes from existing project state, runs experiments, documents everything, and keeps iterating without manual checkpoints unless the user explicitly asks.
---

# Dentist Research Loop

Use this skill for unattended, end-to-end research work in this repository.

This is the Dentist-specific Copilot adaptation of the ARIS loop from:

- `/home/zechuan/Auto-claude-code-research-in-sleep/skills/research-pipeline/SKILL.md`
- `/home/zechuan/Auto-claude-code-research-in-sleep/skills/auto-review-loop/SKILL.md`
- `/home/zechuan/Auto-claude-code-research-in-sleep/skills/research-review/SKILL.md`

The orchestration logic stays the same in spirit, but every external review, novelty-pressure pass, and iteration gate must use the local Copilot custom agent `aris-reviewer` instead of any Codex MCP tool.

## Defaults

- `AUTO_PROCEED = true`
- `HUMAN_CHECKPOINT = false`
- `REVIEWER_AGENT = aris-reviewer`
- `REVIEWER_MODEL = claude-opus-4.6`
- `MAX_REFINEMENT_ROUNDS = 3`
- `MAX_REVIEW_ROUNDS = 4`
- `POSITIVE_STOP = score >= 6/10 OR verdict == READY OR verdict == ALMOST`
- `PRIMARY_LOG = RESEARCH_PIPELINE_REPORT.md`
- `REVIEW_LOG = AUTO_REVIEW.md`
- `STATE_FILE = REVIEW_STATE.json`
- `REFINE_DIR = refine-logs/`

Override these only if the user explicitly asks.

## Non-Negotiable Rules

- Do not use Codex MCP, Codex CLI, or any Codex-specific workflow in this repository.
- Do not stop for approval between stages. The default behavior is full autonomy.
- Prefer resuming the strongest existing direction over inventing a new one.
- Prefer the cheapest decisive experiment package over broad, expensive exploration.
- Persist round state after every major step so the loop survives compaction or interruption.
- Document failures, dead ends, and negative results honestly.

## Project Context to Read First

Always inspect these before choosing the next action:

- `README.md`
- `IDEA_REPORT.md` if present
- `RESEARCH_PIPELINE_REPORT.md` if present
- `AUTO_REVIEW.md` if present
- `REVIEW_STATE.json` if present
- `refine-logs/`
- `scripts/INDEX.md` if present
- the newest summaries in `paper_tables/`
- the newest experiment directories in `runs/`

## High-Level Workflow

### Phase 0: Recover or Start Fresh

1. Check `REVIEW_STATE.json`.
2. If it exists with `"status": "in_progress"` and the timestamp is recent, resume from the next unfinished stage.
3. If it exists with `"status": "completed"` or is stale, start a fresh pass but reuse all prior artifacts as context.
4. Write or refresh a short status header in `RESEARCH_PIPELINE_REPORT.md` describing the current run date, active objective, and whether this is a resume.

### Phase 1: Decide the Active Research Track

Do not restart from idea generation unless the repository truly lacks a clear direction.

Use this priority order:

1. If `refine-logs/FINAL_PROPOSAL.md` exists, treat that as the active method unless recent results clearly invalidate it.
2. Else if `RESEARCH_PIPELINE_REPORT.md` or `IDEA_REPORT.md` already identifies a top idea, continue from that idea.
3. Else infer the strongest active direction from the repository state:
   - current model family and training scripts
   - latest promising metrics
   - documented weaknesses
   - remaining missing evidence for a publishable story

If the direction is still too fuzzy, generate 2-3 grounded candidate directions from the current Dentist codebase and data, then immediately pressure-test them with `aris-reviewer`, auto-select the strongest one, and continue without asking the user.

### Phase 2: Method Refinement

If the active direction is not yet implementation-ready:

1. Create or refresh `refine-logs/round-0-initial-proposal.md` with:
   - problem anchor
   - current bottleneck
   - proposed method thesis
   - minimal experiment package
   - compute/risk estimate
2. Invoke the `aris-reviewer` custom agent with the full proposal and repository context.
3. Save the full raw review to `refine-logs/round-N-review.md`.
4. Revise the method in `refine-logs/round-N-refinement.md`.
5. Repeat up to `MAX_REFINEMENT_ROUNDS` or until the reviewer says the method is concrete and defensible.
6. Write final clean outputs:
   - `refine-logs/FINAL_PROPOSAL.md`
   - `refine-logs/EXPERIMENT_PLAN.md`
   - `refine-logs/REVIEW_SUMMARY.md`

### Phase 3: Implement the Minimum High-Signal Fixes

Once the method is clear:

1. Identify the smallest set of code or experiment changes that most directly address the reviewer’s highest-severity weaknesses.
2. Implement those changes in the local repository.
3. Prefer:
   - metric additions
   - stronger baselines
   - ablations that validate the main claim
   - reframing or result synthesis when that addresses the weakness more cheaply than new training
4. Avoid:
   - broad refactors unrelated to the research claim
   - expensive experiments with weak expected acceptance lift
   - contribution sprawl

### Phase 4: Run and Monitor Experiments

Use the repository’s existing scripts and launch patterns.

1. Reuse or extend existing scripts under `scripts/`.
2. Record commands, configs, and output paths in `RESEARCH_PIPELINE_REPORT.md`.
3. If long experiments are required, launch them in a way that can survive your terminal session and keep monitoring from logs/results directories.
4. When results arrive, summarize the actual evidence in:
   - `RESEARCH_PIPELINE_REPORT.md`
   - `AUTO_REVIEW.md` if inside a review round

### Phase 5: Autonomous Review Loop

Run up to `MAX_REVIEW_ROUNDS` of:

1. Assemble a complete current briefing:
   - active claim
   - method summary
   - exact code/experiment changes since last round
   - latest metrics/tables
   - unresolved risks
2. Invoke `aris-reviewer` with that briefing.
3. Save the raw output verbatim to `AUTO_REVIEW.md`.
4. Parse:
   - score
   - verdict
   - ranked weaknesses
   - minimum fixes
   - claims at risk
5. Stop if `POSITIVE_STOP` is met.
6. Otherwise implement the minimum high-signal fixes and repeat.

For round 2+, include:

- previous score and verdict
- what changed since the last review
- which previous criticisms were addressed
- which criticisms remain open

Do not rely on conversational thread IDs. Each round must be resumable from files alone.

### Phase 6: Finalization

When the loop stops:

1. Update `REVIEW_STATE.json` to `"status": "completed"`.
2. Write a concise final summary to `RESEARCH_PIPELINE_REPORT.md`:
   - active direction
   - implemented changes
   - experiment evidence
   - score progression
   - final verdict
   - remaining blockers, if any
3. Ensure `AUTO_REVIEW.md` includes:
   - all review rounds
   - all raw reviewer outputs
   - actions taken
   - final status

## Required Reviewer Invocation Pattern

Whenever you need external review, use the local Copilot custom agent `aris-reviewer`.

Your prompt to the reviewer must include:

- the active problem statement
- the current method summary
- exact evidence from the latest runs
- known weaknesses
- the question you want answered

Tell the reviewer to return the standard structured format defined in the `aris-reviewer` agent file. Copy the raw response into project documents verbatim before summarizing it.

## Resume Discipline

After every review round or major experiment batch, update `REVIEW_STATE.json` in this shape:

```json
{
  "status": "in_progress",
  "stage": "review_loop",
  "round": 2,
  "active_direction": "one-line description",
  "last_score": 5.5,
  "last_verdict": "NOT READY",
  "pending_runs": [
    "runs/exp_xyz"
  ],
  "timestamp": "2026-03-18T22:00:00+08:00"
}
```

Use `"status": "completed"` only when the current pass is done.

## Dentist-Specific Priorities

- Use existing repository state before proposing new architecture families.
- Prefer advancing the strongest current Dentist story rather than opening a second story branch.
- Always connect any new experiment to a paper-level claim.
- If the latest results are weak, decide explicitly whether to:
  - fix the method,
  - fix the evaluation,
  - narrow the claim,
  - or pivot the framing.

## What Success Looks Like

At the end of a successful run, the repository should contain:

- an updated `RESEARCH_PIPELINE_REPORT.md`
- an updated `AUTO_REVIEW.md`
- an updated `REVIEW_STATE.json`
- refreshed `refine-logs/` documents when method refinement was needed
- concrete code / script / result changes that correspond to the reviewer’s minimum fixes

## Invocation Examples

- `continue the dentist research loop autonomously`
- `run the overnight dentist paper-improvement loop`
- `use dentist-research-loop and keep going without checkpoints`
- `resume the dentist copilot research pipeline from current state`
