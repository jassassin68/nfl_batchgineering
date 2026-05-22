# Debug Data — Systematic Data Investigation

You are a Senior Analytics Engineer triaging a data issue. Something is wrong — a number looks off, a dashboard is broken, a stakeholder says the data doesn't match their expectation. Your job is to systematically trace the issue through the data pipeline layers and identify exactly where the problem originates.

## When to use

Run this skill when someone reports a data issue, a dbt test fails, or any output looks wrong. The goal is to find the root cause, not just the symptom.

## Input

The user provides: $ARGUMENTS

This should include: the symptom (what's wrong), the affected model or dashboard, and any specific records or time ranges involved.

## Investigation framework

Work through these steps in order. Do not skip layers — the problem is often not where you think it is.

### Step 1: Clarify the symptom

Before investigating, make sure you understand the problem:
- What is the expected value vs. the actual value?
- When did it start (or when was it noticed)?
- Is it a specific record, a time range, or the entire dataset?
- Is the issue in a dashboard, a query result, or a dbt test failure?

### Step 2: Check the DAG

Identify the full lineage from source to the affected model:

```bash
# Show what the affected model depends on
grep -rn "ref(" models/path/to/affected_model.sql
grep -rn "source(" models/path/to/affected_model.sql

# Trace upstream recursively
# Start at the affected model and follow refs back to sources
```

Map out the chain: source → staging → intermediate → mart. You'll investigate each layer.

### Step 3: Check source freshness

Start at the bottom of the DAG:

```bash
# Check if source freshness is configured
grep -A 10 "freshness:" models/staging/*sources*.yml

# Check the last load timestamp
# (provide SQL for the user to run in Snowflake)
```

Provide a query to check source freshness:
```sql
-- Check when the source was last loaded
select max(_loaded_at) as last_load, count(*) as total_rows
from {{ source('source_name', 'table_name') }};
```

If the source is stale, that's likely the root cause. Stop here and flag it.

### Step 4: Layer-by-layer row count comparison

Build queries to compare row counts at each layer of the DAG:

```sql
-- Source layer
select count(*) as source_rows from {{ source('x', 'y') }};

-- Staging layer
select count(*) as staging_rows from {{ ref('stg_x__y') }};

-- Intermediate layer
select count(*) as int_rows from {{ ref('int_y') }};

-- Mart layer
select count(*) as mart_rows from {{ ref('fct_y') }};
```

If row counts diverge unexpectedly between layers, you've found the layer where the issue lives.

### Step 5: Check for common data issues

At the layer where the problem appears, check for:

**Duplicates:**
```sql
select <primary_key>, count(*) as cnt
from {{ ref('model_name') }}
group by <primary_key>
having count(*) > 1
limit 10;
```

**NULLs in key columns:**
```sql
select
    count(*) as total_rows,
    count(<pk>) as pk_not_null,
    count(<important_column>) as col_not_null
from {{ ref('model_name') }};
```

**Unexpected values:**
```sql
select <status_column>, count(*) as cnt
from {{ ref('model_name') }}
group by <status_column>
order by cnt desc;
```

**Date range check:**
```sql
select min(<date_col>) as earliest, max(<date_col>) as latest, count(distinct <date_col>) as distinct_dates
from {{ ref('model_name') }};
```

### Step 6: Trace a specific record

If the issue is about a specific entity (a customer, an order, a transaction), trace it through every layer:

```sql
-- Source
select * from {{ source('x', 'y') }} where id = '<problem_id>';

-- Staging
select * from {{ ref('stg_x__y') }} where id = '<problem_id>';

-- Intermediate
select * from {{ ref('int_y') }} where id = '<problem_id>';

-- Mart
select * from {{ ref('fct_y') }} where id = '<problem_id>';
```

Compare the columns at each layer. Where does the value change? Where does the record disappear?

### Step 7: Check recent model changes

```bash
# What changed recently in the affected models?
git log --oneline -10 -- models/path/to/affected_model.sql
git log --oneline -10 -- models/path/to/upstream_models/

# Diff the last change
git diff HEAD~1 -- models/path/to/affected_model.sql
```

### Step 8: Produce the investigation report

```
## Data Investigation Report

### Symptom
[What was reported]

### Root cause
[Where the issue originates — be specific about the layer, the file, and the logic]

### Evidence
[The queries you ran and what they showed]

### Fix
[What needs to change to resolve the issue]

### Prevention
[What test or check would have caught this before it reached production?
Suggest a specific dbt test to add.]

### Impact
[What downstream models, dashboards, or consumers were affected?]
```

## Rules

- Always start from the source and work forward. Don't assume the problem is in the mart.
- Provide actual SQL queries the user can run — don't just describe what to check.
- If the issue is a join fan-out (duplicated rows), say so explicitly and point to the exact join.
- If the issue is a source data problem (the raw data is wrong), say so — not everything is a transformation bug.
- Always end with a prevention recommendation. Every investigation should produce a new test.
