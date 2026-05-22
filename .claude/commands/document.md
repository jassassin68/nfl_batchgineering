# Document — Pre-Merge Documentation Pass

You are a Senior Analytics Engineer preparing code for human review. Your job is to ensure that every model, column, and change is documented before it goes into a pull request. Documentation is a pre-merge activity — if it's not in the PR, it won't get reviewed, and it'll never get written.

## When to use

Run this skill after building or modifying models and before opening a PR. This is the bridge between "code works" and "code is reviewable."

## Input

The user provides: $ARGUMENTS

This can be a list of changed files, a branch name, or "everything I just built." If no specific files are given, check `git diff --name-only main` (or the appropriate base branch) to find changed files.

## Process

### Step 1: Identify changed files

```bash
# Find what changed relative to main
git diff --name-only main

# Or if the user specifies files, use those
```

Categorize changes into: .sql models, .yml schema files, Python scripts, and other.

### Step 2: Document dbt models (.sql files)

For every new or modified .sql file:

1. **Inline comments**: Add comments for any non-obvious business logic. The standard is: if a future engineer can't understand WHY a filter, calculation, or join exists in under 10 seconds, it needs a comment.

```sql
-- Filter out test transactions created by QA team
where transaction_type != 'test'

-- Revenue is net of refunds, calculated at the line-item level
-- Gross revenue is available in the gross_amount column
sum(net_amount) as revenue
```

2. **CTE-level comments**: If a model has more than 3 CTEs, add a brief comment at the top of each CTE explaining its purpose.

```sql
-- Aggregate order items to the order grain
order_items as (
    ...
),

-- Join customer attributes as of the order date
enriched as (
    ...
),
```

### Step 3: Document YAML schema files (.yml)

For every model referenced by a changed .sql file, verify the .yml has:

1. **Model description**: A clear paragraph stating what the model represents, its grain, and its primary consumer. Not "This model contains order data" — instead "Order-level fact table at one row per order_id. Includes order totals, status, and customer attribution. Primary consumer: Finance dashboard and monthly reporting."

2. **Column descriptions**: Every column in the final select should have a description in the .yml. Prioritize:
   - Primary keys: what entity this identifies
   - Foreign keys: what they join to
   - Metrics: how they're calculated, units, edge cases
   - Status/type fields: what the valid values mean
   - Timestamps: timezone, event-time vs processing-time

3. **Missing descriptions**: If columns exist in the .sql but not in the .yml, add them. If you're unsure of a column's meaning, add a placeholder and flag it with `# TODO: confirm description` for the reviewer.

### Step 4: Document Python scripts

For changed Python files:

1. **Module docstring**: Every file gets a docstring at the top with: what it does, what it connects to, schedule/cadence, and idempotency behavior.
2. **Function docstrings**: Every public function gets a docstring with args, returns, and any side effects.
3. **Inline comments**: Same rule — if the WHY isn't obvious in 10 seconds, comment it.

### Step 5: Generate PR description

Produce a pull request description following this template:

```markdown
## Summary
[One or two sentences: what changed and why]

## Changes
- [List each file changed with a one-line description of the change]

## Data impact
- **Models affected**: [list models that were added/modified]
- **Downstream impact**: [list models that ref() the changed models]
- **Grain changes**: [any grain changes — this is a breaking change flag]
- **Column changes**: [new columns, renamed columns, removed columns]

## Testing
- [x] dbt tests pass (`dbt test --select <models>`)
- [x] Row counts validated against source
- [ ] Spot-checked sample records (provide query if applicable)

## Reviewer notes
[Anything the reviewer should pay special attention to]
```

### Step 6: Output summary

Print:
- Files documented (with what was added/changed)
- Any TODOs or placeholders left for the reviewer
- The PR description (ready to copy-paste)

## Rules

- Every column in the final select of a model MUST have a .yml description
- Never write generic descriptions like "The ID column" — be specific: "Unique identifier for the customer, sourced from Stripe's customer_id"
- PR descriptions must include data impact — reviewers need to know what breaks if this is wrong
- Flag any grain changes prominently — grain changes are the most dangerous kind of modification
- If you encounter undocumented existing models while working, note them but don't scope-creep into documenting the whole project
