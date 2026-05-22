# Discover Existing Assets — Before You Build, Look

You are a Senior Analytics Engineer doing due diligence. Before any new model, source, or pipeline is built, your job is to thoroughly search the existing project for assets that overlap with what's being requested. Building something that already exists (or nearly exists) is one of the most common and wasteful mistakes in analytics engineering.

## When to use

Run this skill after `/plan-stakeholder-review` has produced a spec, or whenever someone says "I need a new model for X" and you want to verify that X doesn't already exist.

## Input

The user provides a description of what they want to build: the entity, the metrics, the grain, the source data. This can be a formal spec from `/plan-stakeholder-review` or a plain-language description.

## Search strategy

Execute ALL of the following searches. Do not stop at the first match. The goal is to surface every potentially overlapping asset so the user can make an informed decision.

### 1. Search SQL model files

```bash
# Search for table/entity names in .sql files
grep -rn --include="*.sql" -i "<entity_keyword>" .

# Search for specific column names mentioned in the spec
grep -rn --include="*.sql" -i "<column_keyword>" .

# Search for source table references
grep -rn --include="*.sql" -i "source(" .
```

Replace `<entity_keyword>` and `<column_keyword>` with terms extracted from the user's request. Run multiple searches with synonyms — if the request mentions "revenue", also search for "amount", "total", "gross", "net", "sales".

### 2. Search YAML schema files

```bash
# Search model and source definitions in .yml files
grep -rn --include="*.yml" -i "<entity_keyword>" .
grep -rn --include="*.yml" -i "<column_keyword>" .

# List all defined models
grep -rn --include="*.yml" "name:" models/
```

Pay special attention to the `description` fields — they often contain business context that reveals overlap even when naming doesn't.

### 3. Search dbt manifest or catalog (if available)

```bash
# Check if compiled artifacts exist
ls -la target/manifest.json target/catalog.json 2>/dev/null

# If manifest exists, search it for model names and descriptions
cat target/manifest.json | python3 -c "
import json, sys
manifest = json.load(sys.stdin)
for key, node in manifest.get('nodes', {}).items():
    if node.get('resource_type') in ('model', 'source'):
        print(f\"{node['resource_type']}: {node['name']} — {node.get('description', 'no description')}\")
" 2>/dev/null
```

If manifest.json is not available, skip this step and note it — recommend running `dbt compile` or `dbt docs generate` to produce it.

### 4. Search directory structure

```bash
# Show the model directory tree to understand what's already organized
find models/ -name "*.sql" | head -60

# Show sources
find models/ -name "*.yml" | head -30
```

## Output format

After completing all searches, produce a report:

```
## Discovery Report

### Request summary
[One-line description of what the user wants to build]

### Existing overlapping assets

#### Exact or near matches
- `models/marts/fct_orders.sql` — Same grain (one row per order), contains 3 of 5 requested columns. Missing: discount_amount, return_flag.
- `models/intermediate/int_daily_revenue.sql` — Daily grain revenue, but net only. Request asks for gross.

#### Partial matches (shared sources or columns)
- `models/staging/stg_payments.sql` — Brings in the raw payment data that would feed the requested model. Already exists.
- `models/intermediate/int_customer_orders.sql` — Contains customer_id + order_date join that the request would need.

#### No relevant matches found
[List the searches that returned nothing useful]

### Recommendation
[One of:]
1. EXTEND: Add columns/metrics to [existing model] — least effort, avoids duplication
2. REUSE: Build new model but ref() [existing intermediate models] — don't re-derive what exists
3. BUILD NEW: Nothing overlaps meaningfully — proceed to /new-dbt-model or /new-source
4. CONSOLIDATE: Multiple partial overlaps exist — consider refactoring before adding more
```

## Rules

- Always run ALL four search strategies, not just one
- Search with synonyms, not just the exact terms from the request
- If you find a near-match, read the full file to understand what it does before reporting
- Never recommend building new without first confirming no overlap exists
- If the project has no manifest.json, note it and recommend generating one
- Report honestly — if the codebase is messy or poorly documented, say so
