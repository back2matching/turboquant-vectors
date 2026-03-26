Read CLAUDE.md, ISSUES.md, docs/plans/EXECPLAN.md, and .claude/WORKFLOW.md.

Check git log for what changed since last loop. Run tests. Check for any failing tests or regressions.

Then execute the next uncompleted task from WORKFLOW.md:
1. Read the task description
2. Do the work (code, research, docs — whatever is needed)
3. Run tests to verify nothing broke
4. Update ISSUES.md if you found or fixed bugs
5. Update WORKFLOW.md to mark the task done and add any new tasks discovered
6. Commit your changes with a descriptive message
7. Report what you did and what's next

If all tasks in the current phase are done, advance to the next phase. If all phases are done, run a full audit (read every source file, run all tests, check docs for staleness) and create new tasks based on findings.

Do NOT ask what to do. Read the state, decide, execute. If blocked, document why in ISSUES.md and move to the next task.
