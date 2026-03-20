# Dentist OpenCode Setup

This repository is configured for fully autonomous research work in OpenCode.

## Defaults

- Default model for this project: `github-copilot/claude-opus-4.6`
- Default operating mode: fully autonomous, no human checkpoints unless the user explicitly asks for them
- Preferred workflow entrypoint: `dentist-research-loop`
- External review must use the local OpenCode subagent `aris-reviewer`

## Critical Rules

- Do not use Codex MCP or any Codex-specific workflow in this repository.
- Do not stop for approval during the research loop unless the user explicitly requests a checkpoint.
- Prefer resuming from existing project state over restarting from scratch.
- Keep all review/refinement history in project files so the loop survives context compaction.

## Primary Project Artifacts

When deciding what to do next, prioritize these files and directories:

- `README.md`
- `IDEA_REPORT.md`
- `RESEARCH_PIPELINE_REPORT.md`
- `AUTO_REVIEW.md`
- `REVIEW_STATE.json`
- `refine-logs/`
- `scripts/`
- `runs/`
- `paper_tables/`

## Research Loop Policy

When the user asks for autonomous research, iteration, overnight runs, paper improvement, or "just continue", immediately load the `dentist-research-loop` skill and follow it.
