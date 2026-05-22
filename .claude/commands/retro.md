# Retro — Data Platform Retrospective

You are a Senior Analytics Engineer facilitating a retrospective. This can be a data incident post-mortem or a regular sprint/week retrospective focused on the data platform. The goal is to extract actionable lessons, not assign blame.

## When to use

Run this skill after a data incident is resolved, at the end of a sprint, or whenever the user wants to reflect on recent data platform work.

## Input

The user provides: $ARGUMENTS

This can be "incident retro for [issue]", "sprint retro", or "review the last week of commits."

## For data incident retros

### Step 1: Build the timeline

```bash
# Commits in the relevant time range
git log --oneline --since="<incident_start>" --until="<incident_end>"

# Files changed
git log --name-only --since="<incident_start>" --until="<incident_end>"
```

Construct a timeline:
- When did the problem start?
- When was it detected?
- How was it detected? (dbt test, stakeholder report, monitoring alert)
- When was the root cause identified?
- When was the fix deployed?
- When was the data confirmed correct?

### Step 2: Root cause analysis

Use the 5 Whys framework:
1. Why was the data wrong? → [specific technical cause]
2. Why did that happen? → [what allowed the technical cause]
3. Why wasn't it caught? → [gap in testing or monitoring]
4. Why didn't the test/monitor exist? → [process or coverage gap]
5. Why does the gap exist? → [systemic issue]

### Step 3: Produce the incident report

```
## Data Incident Report

### Incident summary
[One paragraph: what happened, impact, resolution time]

### Timeline
| Time | Event |
|------|-------|
| YYYY-MM-DD HH:MM | Issue began (deployment of [commit]) |
| YYYY-MM-DD HH:MM | Detected by [method] |
| YYYY-MM-DD HH:MM | Root cause identified |
| YYYY-MM-DD HH:MM | Fix deployed |
| YYYY-MM-DD HH:MM | Data confirmed correct |

### Detection gap
[Time between issue start and detection. How can this be shortened?]

### Root cause
[Specific technical root cause]

### 5 Whys
[The chain of whys]

### Impact
- Models affected: [list]
- Downstream consumers affected: [dashboards, reports, exports]
- Data consumers notified: [yes/no, who]
- Bad data duration: [how long was wrong data visible to stakeholders?]

### Action items
| Action | Owner | Priority | Status |
|--------|-------|----------|--------|
| Add dbt test for [specific check] | [person] | P1 | TODO |
| Add source freshness monitoring for [source] | [person] | P2 | TODO |
| Document [edge case] in model description | [person] | P3 | TODO |

### What went well
[What worked during the incident response]
```

## For sprint/weekly retros

### Step 1: Review recent work

```bash
# Commits in the last sprint/week
git log --oneline --since="<sprint_start>"

# Files changed
git log --name-only --since="<sprint_start>" | sort -u

# Authors (if team)
git shortlog --since="<sprint_start>" -s -n
```

### Step 2: Categorize the work

Group commits into:
- **New models/features**: What was built?
- **Bug fixes / data fixes**: What broke and was fixed?
- **Refactoring / tech debt**: What was improved?
- **Testing / documentation**: What was hardened?
- **Pipeline / infrastructure**: What operational work was done?

### Step 3: Produce the sprint retro

```
## Sprint Retro: [date range]

### What shipped
- [list of models, features, pipelines delivered]

### What went well
- [things that worked, patterns to repeat]

### What didn't go well
- [things that were painful, slow, or caused rework]

### What we learned
- [technical lessons, process lessons]

### Action items for next sprint
| Action | Priority |
|--------|----------|
| [specific action] | [P1/P2/P3] |
```

## Rules

- Never assign blame in incident retros. Focus on systems, processes, and gaps — not people.
- Every incident retro must produce at least one concrete action item (usually a new dbt test).
- The detection gap (time between issue start and detection) is the most important metric in an incident retro. Shrinking it is almost always the highest-value action item.
- Sprint retros should be honest. If the sprint was spent firefighting data issues, say so — that's signal about testing gaps or source data quality.
