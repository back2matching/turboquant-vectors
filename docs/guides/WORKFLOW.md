# turboquant-vectors — Development Workflow

> How work gets done. Adapted from FlockRun's ExecPlan system.

---

## The Pipeline

```
Read → Plan → Execute → Test → Docs → Commit
```

### 1. Read
- Read CLAUDE.md for current state
- Check `docs/plans/` for active ExecPlans
- Read relevant source before touching anything

### 2. Plan
For multi-file work, create an ExecPlan in `docs/plans/PLAN-<topic>.md`.

### 3. Execute
Make the change. For parallel work, launch agents for independent file groups.

### 4. Test
```bash
python -m pytest tests/ -v    # All 92 tests must pass
python demos/inversion_demo.py  # Privacy demo (needs sentence-transformers)
```

### 5. Docs (every commit)
| Doc | When to Update |
|-----|---------------|
| CLAUDE.md | If metrics changed (test count, version, features) |
| README.md | If user-facing features or benchmark numbers changed |
| docs/ | If research, plans, or architecture changed |

### 6. Commit
- `type: description` format (feat, fix, docs, refactor)
- Batch related work into one logical commit
- Push to main

---

## ExecPlan System (Autonomous Work)

For multi-hour autonomous sessions, use ExecPlans. These give an agent everything it needs to work without stopping.

### When to Write an ExecPlan
- Multi-file features (3+ files)
- Work expected to take 2+ hours
- Anything needing phased delivery

### ExecPlan Format

Plans live in `docs/plans/`. Name: `PLAN-<topic>.md`.

```markdown
# ExecPlan: <Title>

> One-line description. Created: <date>.

## Purpose
What will exist at the end. 2-3 sentences.

## Progress
Update with timestamps as work proceeds.
- [x] (2026-03-25) Phase 1 done
- [ ] Phase 2 in progress

## Milestones

### Milestone 1: <Name>
**What:** Deliverable description.
**Files:** Paths to create/modify.
**Steps:** Specific steps.
**Verify:** `python -m pytest tests/ -v`
**Done when:** Observable outcome.
```

### ExecPlan Rules
1. **Never stop to ask.** Resolve ambiguities by reading code/docs.
2. **Skip MANUAL items.** Mark BLOCKED, move to next milestone.
3. **Update progress continuously.** Timestamps after each milestone.
4. **Commit at milestones, not steps.**
5. **Verify at each milestone.** Run tests before moving on.

### Completed Plans
Move finished plans to `docs/plans/archive/`.

---

## File Structure

```
docs/
├── README.md           — doc index
├── research/           — specs, threat models, benchmarks, landscape
├── marketing/          — launch plan, Reddit/HN/Twitter drafts
├── plans/              — active ExecPlans
│   └── archive/        — completed plans
└── guides/
    └── WORKFLOW.md     — this file
```

---

## Git Strategy

- **`main`** — single branch, public
- Push after each meaningful milestone
- Tag releases for PyPI publishes

### What's public vs private

| Public (main branch) | Private (.claude/ memory only) |
|----------------------|-------------------------------|
| Source, tests, demos, benchmarks, README | API keys, tokens, credentials |
| docs/ (research, plans, marketing) | Server IPs, SSH keys |
| CLAUDE.md | Internal competitive strategy |

**NEVER commit credentials to any branch.** Those go in `.claude/` memory.

---

## First-Time Audit Prompt

```
Full end-to-end audit of this repo. Read every file — source, tests, docs, config, README, CLAUDE.md, pyproject.toml. Don't skim, actually read the code.

Answer honestly:
1. What does this repo ACTUALLY do?
2. Current state? (version, test count, pass rate, last commit)
3. Finished, active, abandoned, or broken?
4. Docs: what exists, what's missing, what's stale?
5. README accuracy — flag claims not backed by tests
6. What breaks if someone pip installs this right now?

Then fix everything without asking. Update CLAUDE.md, README, docs/. Don't present options. Fix it, commit, move on.

NEVER commit credentials, server IPs, or internal strategy to any branch.
```

---

## Related Projects

| Project | Repo | What |
|---------|------|------|
| FlockRun | github.com/back2matching/flockrun | Parent project (agent runtime) |
| turboquant | github.com/back2matching/turboquant | KV cache compression |
| kvcache-bench | github.com/back2matching/kvcache-bench | KV cache benchmarking |
| quant-sim | github.com/back2matching/quant-sim | Quantization simulator |
