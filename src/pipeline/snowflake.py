"""Centralized Snowflake connectivity for the NFL pipeline.

Single source of truth for key-pair authentication and for resolving the
database/schema where dbt builds its marts. Previously predict.py hardcoded
the database/schema while train_spread_model.py read different env vars, which
made it ambiguous where the marts actually live.

Resolution order for the marts location:
  database: SNOWFLAKE_MARTS_DATABASE -> SNOWFLAKE_DATABASE
  schema:   SNOWFLAKE_MARTS_SCHEMA   -> SNOWFLAKE_SCHEMA
"""

from __future__ import annotations

import os

import polars as pl
from dotenv import load_dotenv


def _private_key_der() -> bytes:
    """Load the key-pair private key from the environment as PKCS8 DER bytes."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    key_text = os.getenv("SNOWFLAKE_KEYPAIR_PRIVATE_KEY")
    if not key_text:
        raise RuntimeError(
            "SNOWFLAKE_KEYPAIR_PRIVATE_KEY is not set in the environment/.env"
        )
    passphrase = os.getenv("SNOWFLAKE_KEYPAIR_PASSPHRASE")
    # .env stores the key on one line with literal \n escapes.
    key = serialization.load_pem_private_key(
        key_text.replace("\\n", "\n").encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def marts_database() -> str:
    """Database where dbt builds its marts."""
    load_dotenv()
    db = os.getenv("SNOWFLAKE_MARTS_DATABASE") or os.getenv("SNOWFLAKE_DATABASE")
    if not db:
        raise RuntimeError(
            "Set SNOWFLAKE_DATABASE (or SNOWFLAKE_MARTS_DATABASE) in .env"
        )
    return db


def marts_schema() -> str:
    """Schema where dbt builds its marts."""
    load_dotenv()
    schema = os.getenv("SNOWFLAKE_MARTS_SCHEMA") or os.getenv("SNOWFLAKE_SCHEMA")
    if not schema:
        raise RuntimeError(
            "Set SNOWFLAKE_SCHEMA (or SNOWFLAKE_MARTS_SCHEMA) in .env"
        )
    return schema


def marts_table(table: str) -> str:
    """Fully-qualified DATABASE.SCHEMA.TABLE identifier for a mart."""
    return f"{marts_database()}.{marts_schema()}.{table}"


def get_connection():
    """Open a Snowflake connection using key-pair authentication."""
    load_dotenv()
    import snowflake.connector

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    if not account or not user:
        raise RuntimeError("SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER must be set in .env")

    return snowflake.connector.connect(
        account=account,
        user=user,
        private_key=_private_key_der(),
        database=marts_database(),
        schema=marts_schema(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )


def query_df(sql: str) -> pl.DataFrame:
    """Run a read query and return the result as a Polars DataFrame.

    Column names are lower-cased to match Python naming conventions. An empty
    result set returns an empty DataFrame with the correct columns.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0].lower() for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    if not rows:
        return pl.DataFrame({col: [] for col in columns})
    return pl.DataFrame(rows, schema=columns, orient="row")
