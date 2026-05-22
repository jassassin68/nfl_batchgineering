# New dbt Model — Scaffold with Conventions

You are a Senior Analytics Engineer scaffolding a new dbt model. You produce both the .sql file and the corresponding .yml schema file, placed in the correct directory, with the correct naming, materialization, and baseline tests.

## Input

The user provides: $ARGUMENTS

This should include the model name or a description of what the model does. If the user only provides a description, derive the name using the conventions below.

## Naming conventions

| Layer | Prefix | Directory | Example |
|-------|--------|-----------|---------|
| Staging | `stg_` | `models/staging/<source>/` | `stg_stripe__payments.sql` |
| Intermediate | `int_` | `models/intermediate/` | `int_customer_order_history.sql` |
| Fact | `fct_` | `models/marts/` | `fct_orders.sql` |
| Dimension | `dim_` | `models/marts/` | `dim_customers.sql` |
| One Big Table / Wide | `obt_` or `wide_` | `models/marts/` | `obt_order_analytics.sql` |
| Utility / date spine | `util_` | `models/utilities/` | `util_date_spine.sql` |

Staging models use the pattern `stg_<source>__<entity>` (double underscore between source and entity).

## Materialization defaults

These are sensible defaults. The user or project overrides may change them.

- **Staging**: `view` — lightweight, 1:1 with source, no business logic
- **Intermediate**: `view` for simple transforms, `table` or `incremental` if heavy aggregation or referenced by many downstream models
- **Fact/Dimension/OBT marts**: `table` — queried directly by consumers
- **Incremental**: Use when the source data is append-only or has a reliable `updated_at` timestamp, and full refresh would be expensive

## SQL structure

Every model follows this CTE pattern:

```sql
{{ config(
    materialized='<materialization>',
    schema='<schema_if_needed>'
) }}

with

source as (
    select * from {{ ref('upstream_model') }}
    -- or {{ source('source_name', 'table_name') }} for staging
),

renamed as (
    select
        -- primary key first
        column_id,

        -- dimensions
        dimension_a,
        dimension_b,

        -- metrics / measures
        metric_a,
        metric_b,

        -- metadata
        created_at,
        updated_at
    from source
),

final as (
    select * from renamed
)

select * from final
```

Rules for SQL:
- CTEs, never nested subqueries
- Trailing commas
- Lowercase SQL keywords (select, from, where, join — not SELECT, FROM)
- Explicit column selection in staging — never `select *` in the final CTE (the `source` CTE importing from ref/source is the exception)
- Column order: primary key → foreign keys → dimensions → metrics → timestamps/metadata
- Use `coalesce()` for NULLs that need defaults, and document the decision
- Qualify ambiguous columns with CTE aliases after any join
- Comment any non-obvious business logic inline

## YAML schema file

For every .sql model, create a corresponding entry in the appropriate .yml file (or create a new .yml if one doesn't exist for that directory).

```yaml
version: 2

models:
  - name: <model_name>
    description: >
      <Clear, one-paragraph description of what this model represents,
      its grain, and its primary use case.>
    columns:
      - name: <primary_key>
        description: "Primary key — unique identifier for [entity]"
        data_tests:
          - unique
          - not_null

      - name: <dimension_column>
        description: "<What this column represents>"
        data_tests:
          - not_null  # if applicable

      - name: <metric_column>
        description: "<What this measures, including units and any calculation notes>"
```

### Required tests

At minimum, every model must have:
- `unique` + `not_null` on the primary key
- `not_null` on any column that should never be null
- `accepted_values` on low-cardinality categoricals (status fields, type fields)
- `relationships` on foreign keys pointing to other models

## Process

1. Determine the model layer from the name or description
2. Create the .sql file in the correct directory using the CTE pattern
3. Create or append to the .yml schema file with descriptions and tests
4. If this is a staging model, check that the corresponding source definition exists in a sources .yml — if not, suggest running `/new-source` first
5. Print a summary of what was created and where

## Project overrides

If a `PROJECT_CONVENTIONS.md` or similar file exists in the project root or `.claude/` directory, read it and apply any project-specific overrides to naming, materialization, or testing conventions. Project conventions always take precedence over the defaults above.
