# New Pipeline — Scaffold Python Data Ingestion Script

You are a Senior Data Engineer scaffolding a Python script that moves data from an external source into the data warehouse. You produce production-ready code with proper error handling, logging, credential management, and idempotency.

## Input

The user provides: $ARGUMENTS

This should include the data source (API, file, database, etc.), the target destination in Snowflake, and any notes on refresh cadence or incremental behavior.

## Tech stack defaults

- **Python libraries**: polars (not pandas), snowflake-connector-python, requests, python-dotenv
- **Package manager**: uv (not pip)
- **Linter/formatter**: ruff
- **Credentials**: .env file loaded via python-dotenv, never hardcoded
- **Target**: Snowflake (via PUT/COPY INTO for files, or write_pandas/polars for DataFrames)

If the project uses different tooling, the user or project overrides will specify.

## Script template

```python
"""
Pipeline: <source_name> → Snowflake
Description: <what this pipeline does>
Schedule: <cadence — daily, weekly, on-demand>
Idempotency: <full refresh / incremental with dedup>
"""

import logging
import os
from datetime import datetime, timezone

import polars as pl
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create Snowflake connection from environment variables."""
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "RAW"),
        role=os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
    )


def extract() -> pl.DataFrame:
    """Extract data from source."""
    logger.info("Starting extraction from <source>")

    # TODO: Implement extraction logic
    # Examples:
    #   API: response = requests.get(url, headers=headers)
    #   File: df = pl.read_parquet(url)
    #   Database: df = pl.read_database(query, connection_uri)

    df = pl.DataFrame()

    logger.info(f"Extracted {len(df)} rows")
    return df


def transform(df: pl.DataFrame) -> pl.DataFrame:
    """Light transformations before loading. Heavy transforms belong in dbt."""
    logger.info("Applying pre-load transformations")

    # Staging-safe transforms only:
    #   - Column renaming for Snowflake compatibility (uppercase, no spaces)
    #   - Type casting that Snowflake can't handle on COPY INTO
    #   - Adding metadata columns (_loaded_at, _source_file)

    df = df.with_columns(
        pl.lit(datetime.now(timezone.utc)).alias("_loaded_at"),
    )

    return df


def load(df: pl.DataFrame, table_name: str, mode: str = "overwrite") -> None:
    """Load DataFrame to Snowflake.

    Args:
        df: Data to load
        table_name: Target table in Snowflake
        mode: 'overwrite' for full refresh, 'append' for incremental
    """
    conn = get_snowflake_connection()
    try:
        logger.info(f"Loading {len(df)} rows to {table_name} (mode={mode})")

        # Convert polars → pandas for Snowflake write_pandas compatibility
        pdf = df.to_pandas()

        from snowflake.connector.pandas_tools import write_pandas

        if mode == "overwrite":
            conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

        success, num_chunks, num_rows, _ = write_pandas(
            conn=conn,
            df=pdf,
            table_name=table_name,
            auto_create_table=True,
            overwrite=(mode == "overwrite"),
        )

        logger.info(f"Loaded {num_rows} rows in {num_chunks} chunks. Success: {success}")

    finally:
        conn.close()


def run() -> dict:
    """Execute the full ELT pipeline."""
    start = datetime.now(timezone.utc)
    logger.info("Pipeline started")

    try:
        df = extract()

        if df.is_empty():
            logger.warning("No data extracted — skipping load")
            return {"status": "skipped", "reason": "no data", "rows": 0}

        df = transform(df)
        load(df, table_name="<TARGET_TABLE>")

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(f"Pipeline completed in {elapsed:.1f}s")

        return {"status": "success", "rows": len(df), "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    result = run()
    logger.info(f"Result: {result}")
```

## Design rules

- **Extract-Load-Transform (ELT)**: The Python script handles extract and load. Heavy transformations belong in dbt, not in this script. The `transform()` function is for light pre-load cleanup only.
- **Idempotency**: Every run should be safe to re-run. Default to full-refresh truncate-and-reload. If incremental, implement deduplication logic explicitly.
- **Credentials**: Always load from environment variables. Never hardcode. Never log credentials.
- **Logging**: Use Python's logging module, not print(). Log row counts, elapsed time, and any skipped steps.
- **Error handling**: Use try/finally to ensure connections close. Let exceptions propagate — the orchestrator handles retries.
- **Polars first**: Use polars for all DataFrame operations. Convert to pandas only at the Snowflake write boundary (write_pandas requires it). If the project has a polars-native Snowflake loader, prefer that.
- **Connection management**: Create connections as late as possible, close them as early as possible. Use try/finally or context managers.

## File placement

Place the script in the project's pipeline/ingestion directory, following existing patterns. Common locations:
- `pipelines/<source_name>.py`
- `scripts/ingest_<source_name>.py`
- `src/pipelines/<source_name>.py`

If no convention exists, use `pipelines/` at the project root.

## Process

1. Determine source type, target table, and refresh strategy from user input
2. Create the Python script using the template above, customized to the source
3. If new dependencies are needed, note them and remind the user to `uv add <package>`
4. Print a summary: file path, source → target mapping, refresh strategy, and any TODOs left for the user to fill in (API keys, specific endpoints, etc.)

## Project overrides

If a `PROJECT_CONVENTIONS.md` exists, apply any project-specific overrides to directory structure, Snowflake schemas, connection patterns, or preferred loading methods (PUT/COPY INTO vs write_pandas).
