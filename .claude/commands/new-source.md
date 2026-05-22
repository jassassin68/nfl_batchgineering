# New Source — Scaffold Source Definition + Staging Model

You are a Senior Analytics Engineer adding a new source to the dbt project. You produce both the source .yml definition and the corresponding staging model, following the convention that every source table gets exactly one staging model that serves as the single entry point for that raw data.

## Input

The user provides: $ARGUMENTS

This should include the source system name and table name, or a description of the raw data being onboarded.

## Source YAML definition

Create or append to `models/staging/<source_name>/_<source_name>__sources.yml`:

```yaml
version: 2

sources:
  - name: <source_name>
    description: >
      <What this source system is and what data it provides.>
    database: "{{ env_var('SNOWFLAKE_DATABASE') }}"  # or hardcode if appropriate
    schema: <raw_schema>
    loader: <fivetran | custom_python | manual | airbyte>
    loaded_at_field: <timestamp_column>  # for freshness checks

    freshness:
      warn_after:
        count: 24
        period: hour
      error_after:
        count: 48
        period: hour

    tables:
      - name: <table_name>
        description: >
          <What this specific table contains, its grain, and notable quirks.>
        columns:
          - name: <primary_key>
            description: "Primary key"
            data_tests:
              - unique
              - not_null
          - name: <column_name>
            description: "<description>"
```

### Source YAML rules

- One sources .yml file per source system, named `_<source_name>__sources.yml`
- The leading underscore is a dbt convention — it sorts config files above model files
- Always include `loaded_at_field` and `freshness` if the source has a reliable timestamp
- If you don't know the exact freshness SLA, default to warn at 24h / error at 48h and note it as a placeholder
- Always include column descriptions for at minimum the primary key and any columns the staging model renames or transforms

## Staging model

Create `models/staging/<source_name>/stg_<source_name>__<table_name>.sql`:

```sql
with

source as (
    select * from {{ source('<source_name>', '<table_name>') }}
),

renamed as (
    select
        -- primary key
        <raw_pk_column> as <clean_pk_name>,

        -- dimensions (rename to project conventions)
        <raw_col> as <clean_col_name>,

        -- timestamps (cast and standardize timezone if needed)
        <raw_timestamp>::timestamp_ntz as <clean_timestamp_name>,

        -- metadata
        _fivetran_synced  -- or equivalent load timestamp

    from source
)

select * from renamed
```

### Staging model rules

- Materialized as `view` — staging models are lightweight and should not store data
- The staging model is the ONLY place that references `{{ source() }}` for this table
- All downstream models use `{{ ref('stg_<source>__<table>') }}`
- Staging is for: renaming columns, casting types, standardizing timestamps, light cleanup
- Staging is NOT for: joins, aggregations, business logic, filtering business rows
- Every column that gets renamed should have the rename documented in the .yml

## Staging YAML schema

Create or append to `models/staging/<source_name>/_<source_name>__models.yml`:

```yaml
version: 2

models:
  - name: stg_<source_name>__<table_name>
    description: >
      Staging model for <source_name>.<table_name>.
      Renames columns to project conventions, casts types, and standardizes timestamps.
      Grain: one row per <entity>.
    columns:
      - name: <primary_key>
        description: "Primary key — <entity> identifier"
        data_tests:
          - unique
          - not_null
```

## Process

1. Determine source system name and table name from user input
2. Create the source directory `models/staging/<source_name>/` if it doesn't exist
3. Create or append to the sources .yml
4. Create the staging .sql model
5. Create or append to the staging models .yml
6. Print a summary: source definition path, staging model path, and what tests are included
7. Remind the user to run `dbt compile` to verify the source is resolvable

## Project overrides

If a `PROJECT_CONVENTIONS.md` exists, apply any project-specific overrides to schema names, database references, freshness thresholds, or naming patterns.
