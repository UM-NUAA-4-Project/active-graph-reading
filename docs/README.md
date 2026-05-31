# Project Docs

This folder contains the canonical collaboration documents for the Active Graph Reading project. Keep these files concise, current, and safe to share with project collaborators.

## Structure

```text
docs/
├── README.md
├── proposal.md
├── roadmap.md
├── decisions.md
├── open-questions.md
├── meeting-notes/
│   └── YYYY-MM-DD-topic.md
└── progress/
    ├── daily/
    │   └── YYYY-MM-DD.md
    └── weekly/
        └── YYYY-Www.md
```

## What Goes Where

- `proposal.md`: the main project proposal and research framing.
- `roadmap.md`: milestone plan, phases, expected deliverables, and target timelines.
- `decisions.md`: append-only record of important technical and project decisions.
- `open-questions.md`: unresolved research, implementation, data, and evaluation questions.
- `meeting-notes/`: collaborator-safe meeting notes, named as `YYYY-MM-DD-topic.md`.
- `progress/daily/`: short daily or every-other-day progress logs.
- `progress/weekly/`: weekly summaries for supervision and paper-writing memory.

## Progress Log Template

Daily progress notes should stay short:

```markdown
# Daily Progress: YYYY-MM-DD

## Done
- ...

## Blockers / Questions
- ...

## Next
- ...
```

Weekly summaries should focus on decisions and momentum:

```markdown
# Weekly Progress: YYYY-Www

## Summary
- ...

## Results
- ...

## Decisions
- ...

## Next Week
- ...
```

## Rules

- Keep private personal notes outside this repo.
- Keep raw data and large generated artifacts out of Git unless explicitly small and useful.
- Put paper-specific reading notes under `notes/papers/` if that folder is created later.
- Put experiment commands, run IDs, and result summaries under `experiments/` or a dedicated result folder once created.
- Use `docs/decisions.md` for project decisions that should be easy to recover later.
