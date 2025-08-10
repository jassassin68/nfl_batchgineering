"""
Improved Training-focused NFL data loader
Optimized for performance, reliability, and maintainability
"""

import os
import sys
import requests
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import polars as pl
import snowflake.connector
from snowflake.connector.errors import DatabaseError, ProgrammingError
from loguru import logger
from dotenv import load_dotenv


# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """Configure comprehensive logging with both file and console output"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler with rotation
    logger.add(
        log_dir / "training_data_load_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


@dataclass
class LoadResult:
    """Result container for load operations"""
    success: bool
    rows_loaded: int
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class TrainingDataLoader:
    """Optimized bulk loader for NFL training data"""
    
    # Dataset URLs - simplified structure
    DATASET_URLS = {
        'play_by_play': 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet',
        'rosters': 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet'
    }
    
    # Snowflake type mapping
    POLARS_TO_SNOWFLAKE = {
        pl.Int8: "NUMBER(3,0)",
        pl.Int16: "NUMBER(5,0)",
        pl.Int32: "NUMBER(10,0)",
        pl.Int64: "NUMBER(19,0)",
        pl.Float32: "FLOAT",
        pl.Float64: "DOUBLE",
        pl.Utf8: "VARCHAR",
        pl.Boolean: "BOOLEAN",
        pl.Date: "DATE",
        pl.Datetime: "TIMESTAMP_NTZ",
        pl.Time: "TIME",
    }
    
    def __init__(self, warehouse: str = None, role: str = None):
        """Initialize with Snowflake connection parameters"""
        self._load_config()
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.role = role or os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        self._conn = None
        self._session = self._create_http_session()
        logger.info("🏈 Training Data Loader initialized")
    
    def _load_config(self) -> None:
        """Load and validate environment configuration"""
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded .env from {env_path}")
        
        # Validate required environment variables
        required_vars = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        logger.debug("✅ All required environment variables present")
    
    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    @property
    def connection(self) -> snowflake.connector.SnowflakeConnection:
        """Lazy connection property with auto-reconnect"""
        if self._conn is None or not self._is_connection_alive():
            self._connect()
        return self._conn
    
    def _is_connection_alive(self) -> bool:
        """Check if the connection is still alive"""
        if self._conn is None:
            return False
        try:
            self._conn.cursor().execute("SELECT 1")
            return True
        except:
            return False
    
    def _connect(self) -> None:
        """Establish Snowflake connection with error handling"""
        try:
            logger.debug("Establishing Snowflake connection...")
            self._conn = snowflake.connector.connect(
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                database="RAW",
                schema="NFLVERSE",
                warehouse=self.warehouse,
                role=self.role,
                session_parameters={
                    'QUERY_TAG': 'NFL_TRAINING_DATA_LOADER',
                    'USE_CACHED_RESULT': False
                }
            )
            
            # Verify connection
            with self._conn.cursor() as cur:
                cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_WAREHOUSE()")
                user, role, database, warehouse = cur.fetchone()
                logger.info(f"✅ Connected: {user}@{database} (Role: {role}, WH: {warehouse})")
                
        except DatabaseError as e:
            logger.error(f"❌ Snowflake connection failed: {e}")
            raise
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for cursor with automatic error handling"""
        cursor = self.connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def download_year_data(self, dataset: str, year: int) -> Optional[pl.DataFrame]:
        """
        Download a single year of data with robust error handling
        
        Returns None if download fails after retries
        """
        url = self.DATASET_URLS[dataset].format(year=year)
        logger.info(f"📥 Downloading {dataset} for {year} from {url}")
        
        try:
            start_time = time.time()
            
            # Use session with retry logic
            response = self._session.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            # Read content in chunks for large files
            content = BytesIO()
            downloaded = 0
            chunk_size = 8192
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    content.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress for large downloads
                    if downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                        logger.debug(f"  Downloaded {downloaded / (1024*1024):.1f}MB...")
            
            content.seek(0)
            
            # Parse parquet file
            df = pl.read_parquet(content)
            
            # Add metadata columns more efficiently
            df = df.with_columns([
                pl.lit(year).alias('source_year').cast(pl.Int32),
                pl.lit('BULK_TRAINING').alias('load_type'),
                pl.lit(datetime.now()).alias('loaded_at').cast(pl.Datetime)
            ])
            
            duration = time.time() - start_time
            size_mb = downloaded / (1024 * 1024)
            rows_per_sec = len(df) / duration if duration > 0 else 0
            
            logger.success(
                f"✅ Downloaded {dataset} {year}: "
                f"{len(df):,} rows, {size_mb:.1f}MB in {duration:.1f}s "
                f"({rows_per_sec:.0f} rows/sec)"
            )
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error downloading {dataset} {year}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to parse {dataset} {year}: {e}", exc_info=True)
            return None
    
    def _clean_dataframe_for_snowflake(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare DataFrame for Snowflake by handling complex types
        More efficient than the original implementation
        """
        logger.debug(f"Cleaning DataFrame with {len(df.columns)} columns, {len(df)} rows")
        
        # Identify columns that need conversion
        columns_to_convert = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            
            # Check for complex types that Snowflake can't handle directly
            if isinstance(dtype, (pl.List, pl.Struct, pl.Object)):
                columns_to_convert.append(col_name)
                logger.debug(f"Converting complex column '{col_name}' ({dtype}) to string")
            # Also check for Null type columns
            elif isinstance(dtype, pl.Null):
                columns_to_convert.append(col_name)
                logger.debug(f"Converting null column '{col_name}' to string")
        
        # Apply conversions if needed
        if columns_to_convert:
            df = df.with_columns([
                pl.col(col).cast(pl.Utf8) for col in columns_to_convert
            ])
            logger.debug(f"Converted {len(columns_to_convert)} complex columns to strings")
        
        # Log sample of data for debugging
        logger.debug(f"Sample data (first row): {df.head(1).to_dicts()}")
        
        return df
    
    def _get_snowflake_type(self, polars_dtype: pl.DataType) -> str:
        """Map Polars data type to Snowflake SQL type"""
        # Direct type mapping
        for pl_type, sf_type in self.POLARS_TO_SNOWFLAKE.items():
            if isinstance(polars_dtype, type(pl_type)):
                return sf_type
        
        # Handle complex types
        if isinstance(polars_dtype, (pl.List, pl.Struct, pl.Object)):
            return "VARIANT"
        
        # Default fallback
        return "VARCHAR"
    
    def load_to_snowflake(self, df: pl.DataFrame, table_name: str, mode: str = "REPLACE") -> LoadResult:
        """
        Load DataFrame to Snowflake using optimized COPY INTO
        
        Args:
            df: Polars DataFrame to load
            table_name: Target table name
            mode: 'REPLACE' to truncate first, 'APPEND' to add to existing
        
        Returns:
            LoadResult with success status and details
        """
        start_time = time.time()
        table_name = table_name.upper()
        
        logger.info(f"📤 Loading {len(df):,} rows to {table_name} (mode: {mode})")
        
        try:
            # Clean the DataFrame
            df_clean = self._clean_dataframe_for_snowflake(df)
            
            # Create temporary parquet file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            
            try:
                # Write parquet file
                df_clean.write_parquet(str(temp_path), compression='snappy')
                file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                logger.debug(f"Created temp file: {temp_path} ({file_size_mb:.1f}MB)")
                
                with self._get_cursor() as cursor:
                    # Create table if needed
                    self._ensure_table_exists(cursor, table_name, df_clean)
                    
                    # Create temporary stage
                    stage_name = f"TEMP_STAGE_{table_name}_{int(time.time())}"
                    cursor.execute(f"CREATE OR REPLACE TEMPORARY STAGE {stage_name}")
                    logger.debug(f"Created temporary stage: {stage_name}")
                    
                    # Upload file to stage (handle Windows paths)
                    file_path = str(temp_path).replace('\\', '/')
                    put_cmd = f"PUT 'file://{file_path}' @{stage_name} AUTO_COMPRESS=FALSE"
                    logger.debug(f"Executing PUT command: {put_cmd}")
                    
                    put_result = cursor.execute(put_cmd).fetchall()
                    logger.debug(f"PUT result: {put_result}")
                    
                    # Verify file was uploaded
                    cursor.execute(f"LIST @{stage_name}")
                    stage_files = cursor.fetchall()
                    logger.debug(f"Files in stage: {stage_files}")
                    
                    if not stage_files:
                        raise RuntimeError(f"File upload failed - no files found in stage {stage_name}")
                    
                    # Handle mode
                    if mode == "REPLACE":
                        cursor.execute(f"TRUNCATE TABLE {table_name}")
                        logger.debug(f"Truncated table {table_name}")
                    
                    # Copy data with error handling
                    copy_cmd = f"""
                        COPY INTO {table_name}
                        FROM @{stage_name}
                        FILE_FORMAT = (TYPE = 'PARQUET')
                        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                        ON_ERROR = 'CONTINUE'
                        PURGE = TRUE
                    """
                    
                    logger.debug(f"Executing COPY command: {copy_cmd}")
                    result = cursor.execute(copy_cmd).fetchall()
                    
                    # Log the raw COPY result for debugging
                    logger.debug(f"COPY result: {result}")
                    
                    # Parse copy results - handle different result formats
                    if result and len(result) > 0:
                        # Result format: (file, status, rows_parsed, rows_loaded, error_limit, errors_seen, ...)
                        rows_loaded = 0
                        rows_error = 0
                        for row in result:
                            if len(row) >= 4:
                                # rows_loaded is typically in position 3 (0-indexed)
                                rows_loaded += int(row[3]) if row[3] else 0
                                # errors_seen is typically in position 5 (0-indexed)
                                if len(row) >= 6:
                                    rows_error += int(row[5]) if row[5] else 0
                    else:
                        # If no result, query the table to check
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        rows_loaded = cursor.fetchone()[0]
                        logger.debug(f"No COPY result returned, table has {rows_loaded} rows")
                    
                    if rows_error > 0:
                        logger.warning(f"⚠️ {rows_error} rows had errors during load")
                        # Query for detailed error information
                        cursor.execute(f"""
                            SELECT * FROM TABLE(VALIDATE({table_name}, JOB_ID => '_last'))
                            LIMIT 10
                        """)
                        errors = cursor.fetchall()
                        if errors:
                            logger.error(f"Sample errors: {errors[:3]}")
                    
                    # Cleanup stage
                    cursor.execute(f"DROP STAGE IF EXISTS {stage_name}")
                    
                    duration = time.time() - start_time
                    
                    logger.success(
                        f"✅ Loaded {rows_loaded:,}/{len(df):,} rows to {table_name} "
                        f"in {duration:.1f}s ({rows_loaded/duration:.0f} rows/sec)"
                    )
                    
                    return LoadResult(
                        success=True,
                        rows_loaded=rows_loaded,
                        duration_seconds=duration
                    )
                    
            finally:
                # Always cleanup temp file
                if temp_path.exists():
                    temp_path.unlink()
                    logger.debug("Cleaned up temporary file")
                    
        except Exception as e:
            error_msg = f"Failed to load {table_name}: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return LoadResult(
                success=False,
                rows_loaded=0,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )
    
    def _ensure_table_exists(self, cursor, table_name: str, sample_df: pl.DataFrame) -> None:
        """Create table if it doesn't exist with proper schema"""
        # Check existence
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'NFLVERSE' 
            AND TABLE_NAME = '{table_name}'
        """)
        
        if cursor.fetchone()[0] > 0:
            logger.debug(f"Table {table_name} already exists")
            return
        
        # Generate CREATE TABLE DDL
        columns = []
        for col_name in sample_df.columns:
            sf_type = self._get_snowflake_type(sample_df[col_name].dtype)
            columns.append(f'"{col_name}" {sf_type}')
        
        create_ddl = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
        """
        
        cursor.execute(create_ddl)
        logger.info(f"✅ Created table {table_name} with {len(columns)} columns")
    
    def bulk_load_dataset(self, dataset: str, years: List[int]) -> LoadResult:
        """
        Bulk load multiple years of a dataset
        
        Optimized to combine years before loading
        """
        logger.info(f"🔄 Starting bulk load: {dataset} for years {years}")
        start_time = time.time()
        
        # Download all years in parallel (could be further optimized with threading)
        dataframes = []
        failed_years = []
        
        for year in years:
            df = self.download_year_data(dataset, year)
            if df is not None:
                dataframes.append(df)
            else:
                failed_years.append(year)
        
        if not dataframes:
            error_msg = f"No data successfully downloaded for {dataset}"
            logger.error(f"❌ {error_msg}")
            return LoadResult(success=False, rows_loaded=0, error_message=error_msg)
        
        if failed_years:
            logger.warning(f"⚠️ Failed to download {dataset} for years: {failed_years}")
        
        # Combine all DataFrames efficiently
        logger.info(f"🔗 Combining {len(dataframes)} years of data...")
        combined_df = pl.concat(dataframes, how="vertical")
        
        logger.info(
            f"📊 Combined dataset: {len(combined_df):,} rows, "
            f"{len(combined_df.columns)} columns, "
            f"{combined_df.estimated_size() / (1024**2):.1f}MB"
        )
        
        # Load to Snowflake
        result = self.load_to_snowflake(combined_df, dataset)
        result.duration_seconds = time.time() - start_time
        
        return result
    
    def load_all_training_data(self, years: List[int] = None) -> Dict[str, LoadResult]:
        """
        Main entry point to load all training datasets
        
        Args:
            years: List of years to load (default: [2023, 2024])
        
        Returns:
            Dictionary of dataset names to LoadResult objects
        """
        if years is None:
            years = [2023, 2024]
        
        logger.info(f"🚀 Starting bulk training data load for years: {years}")
        logger.info(f"📦 Datasets to load: {list(self.DATASET_URLS.keys())}")
        
        results = {}
        total_start = time.time()
        
        # Load each dataset
        for i, dataset in enumerate(self.DATASET_URLS.keys(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Dataset {i}/{len(self.DATASET_URLS)}: {dataset}")
            logger.info(f"{'='*60}")
            
            results[dataset] = self.bulk_load_dataset(dataset, years)
        
        # Summary statistics
        total_duration = time.time() - total_start
        successful = sum(1 for r in results.values() if r.success)
        total_rows = sum(r.rows_loaded for r in results.values())
        
        logger.info(f"\n{'='*60}")
        logger.info("📈 LOAD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful: {successful}/{len(results)} datasets")
        logger.info(f"📊 Total rows: {total_rows:,}")
        logger.info(f"⏱️ Total time: {total_duration:.1f}s")
        logger.info(f"🚀 Throughput: {total_rows/total_duration:.0f} rows/sec")
        
        if successful < len(results):
            failed = [name for name, result in results.items() if not result.success]
            logger.warning(f"❌ Failed datasets: {failed}")
        else:
            logger.success("🎉 All datasets loaded successfully!")
        
        return results
    
    def validate_data(self) -> Dict[str, Dict]:
        """Validate loaded data with comprehensive checks"""
        logger.info("🔍 Validating loaded data...")
        
        validation_results = {}
        
        with self._get_cursor() as cursor:
            # Get all tables in schema
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'NFLVERSE'
                ORDER BY TABLE_NAME
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                try:
                    # Comprehensive validation query - Snowflake compatible
                    cursor.execute(f"""
                        SELECT 
                            COUNT(*) as row_count,
                            SUM(CASE WHEN source_year IS NULL THEN 1 ELSE 0 END) as missing_year,
                            MIN(source_year) as min_year,
                            MAX(source_year) as max_year,
                            COUNT(DISTINCT source_year) as unique_years,
                            MAX(loaded_at) as last_load_time
                        FROM {table}
                    """)
                    
                    row = cursor.fetchone()
                    
                    validation_results[table] = {
                        'row_count': row[0],
                        'missing_year_count': row[1] or 0,
                        'year_range': f"{row[2]}-{row[3]}" if row[2] else "N/A",
                        'unique_years': row[4] or 0,
                        'last_loaded': row[5].strftime('%Y-%m-%d %H:%M:%S') if row[5] else "Unknown",
                        'status': '✅' if row[0] > 0 else '⚠️'
                    }
                    
                    logger.info(
                        f"{validation_results[table]['status']} {table}: "
                        f"{row[0]:,} rows, "
                        f"{row[4] or 0} years ({row[2]}-{row[3]})"
                    )
                    
                except Exception as e:
                    validation_results[table] = {
                        'status': '❌',
                        'error': str(e)
                    }
                    logger.error(f"❌ {table}: Validation failed - {e}")
        
        return validation_results
    
    def close(self):
        """Clean up resources"""
        if self._conn:
            try:
                self._conn.close()
                logger.info("🔌 Closed Snowflake connection")
            except:
                pass
        
        if self._session:
            self._session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def main():
    """Main execution entry point"""
    setup_logging("DEBUG")  # Changed to DEBUG for more visibility
    
    logger.info("🏈 NFL Training Data Loader Starting")
    logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Use context manager for automatic cleanup
        with TrainingDataLoader() as loader:
            # Configure years to load
            training_years = [2024]  # Start with just one year for testing
            # training_years = list(range(2019, 2025))  # Full dataset
            
            logger.info(f"🎯 Loading data for years: {training_years}")
            
            # Load all data
            results = loader.load_all_training_data(training_years)
            
            # Validate
            validation = loader.validate_data()
            
            # Print detailed results
            logger.info("\n📊 DETAILED RESULTS:")
            for dataset, result in results.items():
                if result.success:
                    logger.info(f"  ✅ {dataset}: {result.rows_loaded:,} rows in {result.duration_seconds:.1f}s")
                else:
                    logger.error(f"  ❌ {dataset}: {result.error_message}")
            
            # Final status
            all_successful = all(r.success for r in results.values())
            
            if all_successful:
                logger.success("🎉 All training data loaded successfully!")
                logger.info("📋 Next steps: Run dbt staging models")
                return 0
            else:
                logger.warning("⚠️ Some datasets failed to load")
                return 1
                
    except KeyboardInterrupt:
        logger.warning("⚡ Load interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())