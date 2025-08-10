# src/ingestion/training_data_loader.py
"""
Training-focused NFL data loader
Simple bulk loading for model training - no incremental complexity
Uses polars + DuckDB for high performance data processing
"""

import os
import requests
from io import BytesIO
from typing import Dict, List
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

import polars as pl
import duckdb
import snowflake.connector
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
    print("✅ .env loaded successfully")
    
    # Check key variables
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    user = os.getenv('SNOWFLAKE_USER')
    
    print(f"SNOWFLAKE_ACCOUNT: {'✅ Found' if account else '❌ Missing'}")
    print(f"SNOWFLAKE_USER: {'✅ Found' if user else '❌ Missing'}")
else:
    print("❌ .env file not found")


class TrainingDataLoader:
    """Simple bulk loader for training data - no incremental logic"""
    
    # nflverse GitHub URLs for bulk downloads
    DATASET_URLS = {
        'play_by_play': 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet', 
        'rosters': 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet'
        #'play_by_play_participation': 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet', 
        #'player_stats': 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet',
        #'injuries': 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet',
    }
    
    def __init__(self):
        """Initialize Snowflake connection for training data loading"""
        self.conn = snowflake.connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"), 
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database="RAW",
            schema="NFLVERSE",
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            role=os.getenv("SNOWFLAKE_ROLE")
        )
        logger.info("🏈 Training Data Loader initialized")
        
        # Test connection
        cursor = self.conn.cursor()
        cursor.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE()")
        user, role, database = cursor.fetchone()
        logger.info(f"✅ Connected as {user} with role {role} to database {database}")
    
    def download_year_data(self, dataset: str, year: int) -> pl.DataFrame:
        """Download a single year of data for a dataset"""
        url = self.DATASET_URLS[dataset].format(year=year)
        logger.info(f"📥 Downloading {dataset} for {year}...")
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=600)  # 10 minute timeout for large files
            response.raise_for_status()
            download_time = time.time() - start_time
            
            # Read parquet from downloaded bytes
            df = pl.read_parquet(BytesIO(response.content))
            
            # Add metadata columns
            df = df.with_columns([
                pl.lit(year).alias('source_year'),
                pl.lit('BULK_TRAINING').alias('load_type'),
                pl.lit(datetime.now()).alias('loaded_at')
            ])
            
            file_size_mb = len(response.content) / (1024 * 1024)
            logger.info(f"✅ Downloaded {len(df):,} rows for {dataset} {year} ({file_size_mb:.1f}MB in {download_time:.1f}s)")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {dataset} {year}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to parse {dataset} {year}: {e}")
            raise
    
    # def download_schedules_all_years(self) -> pl.DataFrame:
    #     """Download schedules (all years in one file)"""
    #     url = self.DATASET_URLS['schedules']
    #     logger.info(f"📥 Downloading all schedules...")
        
    #     try:
    #         start_time = time.time()
    #         response = requests.get(url, timeout=300)
    #         response.raise_for_status()
    #         download_time = time.time() - start_time
            
    #         df = pl.read_parquet(BytesIO(response.content))
            
    #         # Add metadata columns
    #         df = df.with_columns([
    #             pl.lit('BULK_TRAINING').alias('load_type'),
    #             pl.lit(pl.datetime.now()).alias('loaded_at')
    #         ])
            
    #         file_size_mb = len(response.content) / (1024 * 1024)
    #         logger.info(f"✅ Downloaded {len(df):,} schedule records ({file_size_mb:.1f}MB in {download_time:.1f}s)")
    #         return df
            
    #     except Exception as e:
    #         logger.error(f"❌ Failed to download schedules: {e}")
    #         raise
    
    def load_to_snowflake(self, df: pl.DataFrame, table_name: str) -> bool:
        """Load polars DataFrame to Snowflake table using efficient COPY INTO method"""
        try:
            logger.info(f"📤 Loading {len(df):,} rows to {table_name}...")
            start_time = time.time()
            
            # Create temporary parquet file for efficient loading
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                temp_file_path = tmp_file.name
            
            # Convert Windows path separators for Snowflake
            temp_file_path_snowflake = temp_file_path.replace('\\', '/')
            
            try:
                # Clean the DataFrame for Snowflake compatibility
                df_clean = self._prepare_df_for_snowflake(df)
                
                # Write to parquet using polars (much faster than pandas conversion)
                df_clean.write_parquet(temp_file_path)
                logger.debug(f"📁 Created temporary file: {temp_file_path}")
                
                cursor = self.conn.cursor()
                
                # Create table if it doesn't exist (infer from DataFrame schema)
                self._create_table_if_not_exists(table_name, df_clean)
                
                # Create temporary stage for file upload
                stage_name = f"temp_stage_{table_name}_{int(time.time())}"
                cursor.execute(f"CREATE OR REPLACE TEMPORARY STAGE {stage_name}")
                
                # Upload file to stage - use forward slashes for Snowflake
                put_command = f"PUT 'file://{temp_file_path_snowflake}' @{stage_name}"
                logger.debug(f"📤 Executing: {put_command}")
                cursor.execute(put_command)
                
                # Clear existing data (REPLACE mode for training)
                cursor.execute(f"TRUNCATE TABLE {table_name.upper()}")
                
                # Copy from stage to table with better parquet handling
                copy_command = f"""
                    COPY INTO {table_name.upper()}
                    FROM @{stage_name}
                    FILE_FORMAT = (
                        TYPE = 'PARQUET'
                        BINARY_AS_TEXT = FALSE
                    )
                    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                    ON_ERROR = 'CONTINUE'
                """
                logger.debug(f"📋 Executing COPY command: {copy_command}")
                cursor.execute(copy_command)
                
                # Get row count for confirmation
                cursor.execute(f"SELECT COUNT(*) FROM {table_name.upper()}")
                loaded_rows = cursor.fetchone()[0]
                
                # Cleanup stage
                cursor.execute(f"DROP STAGE {stage_name}")
                
                load_time = time.time() - start_time
                logger.info(f"✅ Successfully loaded {loaded_rows:,} rows to {table_name} in {load_time:.1f}s")
                return True
                
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            logger.error(f"❌ Error loading {table_name}: {e}")
            return False
    
    def _prepare_df_for_snowflake(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare DataFrame for Snowflake by handling complex data types"""
        try:
            # Get schema info
            schema = df.schema
            
            # Convert problematic types to strings
            expressions = []
            
            for col_name, dtype in schema.items():
                dtype_str = str(dtype)
                
                # Handle complex types that cause issues
                if any(problem_type in dtype_str.lower() for problem_type in ['list', 'struct', 'array', 'object']):
                    logger.debug(f"Converting complex column {col_name} ({dtype_str}) to string")
                    expressions.append(pl.col(col_name).cast(pl.Utf8).alias(col_name))
                elif 'null' in dtype_str.lower():
                    # Handle null columns
                    logger.debug(f"Converting null column {col_name} to string")
                    expressions.append(pl.col(col_name).cast(pl.Utf8).alias(col_name))
                else:
                    # Keep the column as-is
                    expressions.append(pl.col(col_name))
            
            if expressions:
                df_clean = df.select(expressions)
                logger.debug(f"📊 Cleaned DataFrame: {len(df_clean.columns)} columns")
                return df_clean
            else:
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Could not clean DataFrame: {e}, using original")
            return df
    
    
    def _create_table_if_not_exists(self, table_name: str, sample_df: pl.DataFrame) -> None:
        """Create Snowflake table based on polars DataFrame schema"""
        try:
            cursor = self.conn.cursor()
            
            # Check if table exists
            cursor.execute(f"""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'NFLVERSE'
                AND TABLE_NAME = '{table_name.upper()}'
                AND TABLE_CATALOG = CURRENT_DATABASE()
            """)
            
            if cursor.fetchone()[0] > 0:
                logger.debug(f"Table {table_name} already exists")
                return
            
            # Generate CREATE TABLE statement from polars schema
            columns = []
            for name, dtype in sample_df.schema.items():
                snowflake_type = self._polars_to_snowflake_type(dtype)
                columns.append(f'"{name}" {snowflake_type}')
            
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name.upper()} (
                    {', '.join(columns)}
                )
            """
            
            cursor.execute(create_sql)
            logger.info(f"✅ Created table {table_name} with {len(columns)} columns")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not auto-create table {table_name}: {e}")
            logger.info("Table will be created automatically by COPY INTO command")
    
    def _polars_to_snowflake_type(self, polars_type) -> str:
        """Convert polars data type to Snowflake SQL type"""
        # Handle the polars type object properly
        type_str = str(polars_type)
        
        if 'Int64' in type_str or 'Int32' in type_str:
            return "NUMBER(38,0)"
        elif 'Float64' in type_str or 'Float32' in type_str:
            return "FLOAT"
        elif 'Utf8' in type_str or 'String' in type_str:
            return "VARCHAR"
        elif 'Boolean' in type_str:
            return "BOOLEAN"
        elif 'Date' in type_str:
            return "DATE"
        elif 'Datetime' in type_str:
            return "TIMESTAMP_NTZ"
        elif 'Time' in type_str:
            return "TIME"
        else:
            return "VARCHAR"  # Default fallback
    
    def bulk_load_dataset(self, dataset: str, years: List[int]) -> bool:
        """Bulk load multiple years of a dataset"""
        logger.info(f"🔄 Bulk loading {dataset} for years: {years}")
        
        all_data = []
        failed_years = []
        
        # Download each year
        for year in years:
            try:
                df = self.download_year_data(dataset, year)
                all_data.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Skipping {dataset} {year}: {e}")
                failed_years.append(year)
                continue
        
        if not all_data:
            logger.error(f"❌ No data downloaded for {dataset}")
            return False
        
        if failed_years:
            logger.warning(f"⚠️ Failed to load {dataset} for years: {failed_years}")
        
        # Combine all years
        logger.info(f"🔗 Combining {len(all_data)} years of {dataset} data...")
        combined_df = pl.concat(all_data)
        logger.info(f"📊 Combined dataset: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
        
        # Load to Snowflake
        success = self.load_to_snowflake(combined_df, dataset)
        
        if success:
            logger.info(f"🎉 Successfully bulk loaded {dataset}")
        
        return success
    
    # def bulk_load_schedules(self) -> bool:
    #     """Load schedules (special case - all years in one file)"""
    #     logger.info(f"🔄 Loading schedules (all years)")
        
    #     try:
    #         df = self.download_schedules_all_years()
            
    #         # Filter to training years if needed (optional - can load all)
    #         # Uncomment next line to limit to specific years:
    #         # df = df.filter(pl.col('season') >= 2019)
            
    #         success = self.load_to_snowflake(df, 'schedules')
    #         return success
            
    #     except Exception as e:
    #         logger.error(f"❌ Failed to load schedules: {e}")
    #         return False
    
    def load_all_training_data(self, years: List[int] = None) -> Dict[str, bool]:
        """Load all datasets for training"""
        if years is None:
            years = [2023, 2024]  # Default training range
        
        logger.info(f"🚀 Starting bulk training data load for years: {years}")
        logger.info(f"⏰ Estimated time: 10-15 minutes depending on internet speed")
        
        results = {}
        total_start_time = time.time()
        
        # Load schedules first (doesn't depend on years)
        # logger.info("📅 Loading schedules...")
        # results['schedules'] = self.bulk_load_schedules()
        
        # Load year-based datasets
        datasets_to_load = ['rosters', 'play_by_play']
        
        for i, dataset in enumerate(datasets_to_load, 1):
            logger.info(f"📊 Loading {dataset} ({i}/{len(datasets_to_load)})...")
            results[dataset] = self.bulk_load_dataset(dataset, years)
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        total_time = time.time() - total_start_time
        
        logger.info(f"📈 Training data load complete!")
        logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
        logger.info(f"✅ Success: {successful}/{total} datasets loaded successfully")
        
        if successful == total:
            logger.info("🎉 All training data loaded successfully!")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"⚠️ Failed datasets: {failed}")
        
        return results
    
    def validate_training_data(self) -> Dict[str, Dict]:
        """Validate loaded training data using efficient SQL queries"""
        logger.info("🔍 Validating training data...")
        
        validation_results = {}
        
        for table in ['PLAY_BY_PLAY', 'ROSTERS']:
            try:
                cursor = self.conn.cursor()
                
                # Single efficient query for all validation metrics
                validation_query = f"""
                SELECT 
                    COUNT(*) as row_count,
                    COUNT(DISTINCT season) as season_count,
                    MIN(season) as min_season,
                    MAX(season) as max_season
                FROM {table}
                """
                
                cursor.execute(validation_query)
                row_count, season_count, min_season, max_season = cursor.fetchone()
                
                validation_results[table] = {
                    'row_count': row_count,
                    'season_count': season_count,
                    'season_range': f"{min_season}-{max_season}",
                    'status': '✅' if row_count > 0 else '❌'
                }
                
                logger.info(f"{validation_results[table]['status']} {table}: {row_count:,} rows, {season_count} seasons ({min_season}-{max_season})")
                
            except Exception as e:
                validation_results[table] = {
                    'error': str(e),
                    'status': '❌'
                }
                logger.error(f"❌ {table}: Validation failed - {e}")
        
        return validation_results
    
    def get_data_summary_with_duckdb(self) -> None:
        """Print comprehensive summary of loaded training data"""
        logger.info("📊 Training Data Summary:")
        validation = self.validate_training_data()
        
        # Calculate totals
        total_rows = sum(v.get('row_count', 0) for v in validation.values() if isinstance(v.get('row_count'), int))
        successful_tables = sum(1 for v in validation.values() if v.get('status') == '✅')
        
        logger.info(f"📈 Total rows across all tables: {total_rows:,}")
        logger.info(f"📋 Successfully loaded tables: {successful_tables}/{len(validation)}")
        
        # Additional summary statistics
        # if successful_tables > 0:
        #     try:
        #         cursor = self.conn.cursor()
                
        #         # Get some interesting stats
        #         # cursor.execute("""
        #         #     SELECT 
        #         #         COUNT(DISTINCT season) as seasons,
        #         #         MIN(season) as first_season,
        #         #         MAX(season) as last_season
        #         #     FROM SCHEDULES
        #         # """)
        #         # seasons, first_season, last_season = cursor.fetchone()
        #         # logger.info(f"🗓️  Season coverage: {seasons} seasons ({first_season}-{last_season})")
                
        #         cursor.execute("SELECT COUNT(DISTINCT game_id) FROM SCHEDULES")
        #         total_games = cursor.fetchone()[0]
        #         logger.info(f"🏈 Total games: {total_games:,}")
                
        #         cursor.execute("SELECT COUNT(DISTINCT player_id) FROM PLAYER_STATS")
        #         total_players = cursor.fetchone()[0]
        #         logger.info(f"👥 Total players: {total_players:,}")
                
        #     except Exception as e:
        #         logger.warning(f"⚠️ Could not generate additional stats: {e}")
    
    def close(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            logger.info("🔌 Closed Snowflake connection")


def main():
    """Main execution for training data loading"""
    
    # Set up logging with file output
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "training_data_load_{time}.log", 
        rotation="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("🚀 Starting NFL Training Data Loader")
    
    try:
        loader = TrainingDataLoader()
        
        # Load training data
        # Start with recent years for faster testing, expand as needed
        training_years = [2023, 2024]  # Start small for testing
        # training_years = [2019, 2020, 2021, 2022, 2023, 2024]  # Full dataset
        
        logger.info(f"📋 Loading data for years: {training_years}")
        results = loader.load_all_training_data(training_years)
        
        # Validate results
        loader.get_data_summary_with_duckdb()
        
        # Close connection
        loader.close()
        
        if all(results.values()):
            logger.info("🎉 Training data setup complete! Ready for dbt staging models.")
            print("\n" + "="*60)
            print("✅ SUCCESS: Training data loaded successfully!")
            print("📋 Next step: Set up dbt staging models")
            print("="*60)
        else:
            logger.warning("⚠️ Some datasets failed to load. Check logs for details.")
            failed_datasets = [k for k, v in results.items() if not v]
            print(f"\n❌ Failed datasets: {failed_datasets}")
            print("📋 Check logs for error details")
            
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        print(f"\n❌ FAILED: {e}")
        print("📋 Check logs for full error details")
        raise


if __name__ == "__main__":
    main()


# Example usage:
# python src/ingestion/training_data_loader.py
#
# Or programmatically:
# from src.ingestion.training_data_loader import TrainingDataLoader
# loader = TrainingDataLoader()
# loader.load_all_training_data([2023, 2024])  # Start with recent years
# loader.get_data_summary_with_duckdb()
# loader.close()