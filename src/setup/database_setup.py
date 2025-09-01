#!/usr/bin/env python3
"""
Database Setup Script for NFL Prediction System
Executes the complete Snowflake database setup and validates configuration
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import snowflake.connector
from snowflake.connector import DictCursor
import logging
from datetime import datetime
from dotenv import load_dotenv


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/database_setup.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Handles Snowflake database setup and validation"""
    
    def _load_config(self) -> None:
        """Load and validate environment configuration"""
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded .env from {env_path}")
    
    def __init__(self):
        """Initialize with connection parameters from environment"""
        self._load_config()
        self.connection_params = {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            'role': os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN'),
            'database': 'NFL_BATCHGINEERING'
        }
        
        # Validate required environment variables
        required_vars = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        self.conn = None
        
    def connect(self) -> None:
        """Establish connection to Snowflake"""
        try:
            logger.info("Connecting to Snowflake...")
            self.conn = snowflake.connector.connect(**self.connection_params)
            logger.info("Successfully connected to Snowflake")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from Snowflake")
    
    def execute_sql_file(self, sql_file_path: Path) -> List[Dict[str, Any]]:
        """Execute SQL commands from file"""
        if not self.conn:
            raise RuntimeError("Not connected to Snowflake")
        
        logger.info(f"Executing SQL file: {sql_file_path}")
        
        # Read SQL file
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        # Split into individual statements (basic splitting on semicolon)
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        results = []
        cursor = self.conn.cursor(DictCursor)
        
        try:
            for i, statement in enumerate(statements):
                if statement.upper().startswith(('SELECT', 'SHOW', 'DESCRIBE')):
                    logger.info(f"Executing query statement {i+1}/{len(statements)}")
                    cursor.execute(statement)
                    result = cursor.fetchall()
                    results.append({
                        'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                        'result': result,
                        'row_count': len(result) if result else 0
                    })
                else:
                    logger.info(f"Executing DDL statement {i+1}/{len(statements)}")
                    cursor.execute(statement)
                    results.append({
                        'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                        'result': 'Success',
                        'row_count': cursor.rowcount
                    })
                    
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise
        finally:
            cursor.close()
        
        logger.info(f"Successfully executed {len(statements)} SQL statements")
        return results
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that the database setup was successful"""
        if not self.conn:
            raise RuntimeError("Not connected to Snowflake")
        
        logger.info("Validating database setup...")
        validation_results = {}
        cursor = self.conn.cursor(DictCursor)
        
        try:
            # Check schemas exist
            cursor.execute("SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING")
            schemas = cursor.fetchall()
            schema_names = [schema['name'] for schema in schemas]
            
            required_schemas = ['RAW', 'STAGING', 'INTERMEDIATE', 'MARTS', 'ML']
            missing_schemas = [schema for schema in required_schemas if schema not in schema_names]
            
            validation_results['schemas'] = {
                'required': required_schemas,
                'found': schema_names,
                'missing': missing_schemas,
                'status': '✅' if not missing_schemas else '❌'
            }
            
            # Check warehouses exist
            cursor.execute("SHOW WAREHOUSES LIKE '%WH'")
            warehouses = cursor.fetchall()
            warehouse_names = [wh['name'] for wh in warehouses]
            
            required_warehouses = ['COMPUTE_WH', 'ML_TRAINING_WH']
            missing_warehouses = [wh for wh in required_warehouses if wh not in warehouse_names]
            
            validation_results['warehouses'] = {
                'required': required_warehouses,
                'found': warehouse_names,
                'missing': missing_warehouses,
                'status': '✅' if not missing_warehouses else '❌'
            }
            
            # Check RAW.NFLVERSE schema exists (tables created by TrainingDataLoader)
            cursor.execute("USE SCHEMA RAW.NFLVERSE")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_names = [table['name'] for table in tables]
            
            # Note: Tables are created dynamically by TrainingDataLoader, not by setup
            expected_tables = [
                'PLAY_BY_PLAY', 'PLAYER_SUMMARY_STATS', 'TEAM_SUMMARY_STATS',
                'ROSTERS', 'PLAY_BY_PLAY_PARTICIPATION', 'INJURIES'
            ]
            
            validation_results['raw_tables'] = {
                'expected_after_data_load': expected_tables,
                'currently_found': table_names,
                'note': 'Tables created by TrainingDataLoader, not database setup',
                'status': '✅'  # Always pass since tables are created later
            }
            
            # Check ML schema tables
            cursor.execute("USE SCHEMA ML")
            cursor.execute("SHOW TABLES")
            ml_tables = cursor.fetchall()
            ml_table_names = [table['name'] for table in ml_tables]
            
            required_ml_tables = ['PREDICTIONS', 'MODEL_ARTIFACTS', 'MODEL_PERFORMANCE']
            missing_ml_tables = [table for table in required_ml_tables if table not in ml_table_names]
            
            validation_results['ml_tables'] = {
                'required': required_ml_tables,
                'found': ml_table_names,
                'missing': missing_ml_tables,
                'status': '✅' if not missing_ml_tables else '❌'
            }
            
            # Check file formats
            cursor.execute("USE SCHEMA RAW")
            cursor.execute("SHOW FILE FORMATS")
            file_formats = cursor.fetchall()
            format_names = [fmt['name'] for fmt in file_formats]
            
            required_formats = ['PARQUET_FORMAT', 'CSV_FORMAT', 'JSON_FORMAT']
            missing_formats = [fmt for fmt in required_formats if fmt not in format_names]
            
            validation_results['file_formats'] = {
                'required': required_formats,
                'found': format_names,
                'missing': missing_formats,
                'status': '✅' if not missing_formats else '❌'
            }
            
            # Check resource monitors
            cursor.execute("SHOW RESOURCE MONITORS")
            monitors = cursor.fetchall()
            monitor_names = [monitor['name'] for monitor in monitors]
            
            validation_results['resource_monitors'] = {
                'found': monitor_names,
                'status': '✅' if 'NFL_DAILY_MONITOR' in monitor_names else '❌'
            }
            
            # Overall status
            all_statuses = [result['status'] for result in validation_results.values()]
            validation_results['overall_status'] = '✅' if all(status == '✅' for status in all_statuses) else '❌'
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results['error'] = str(e)
            validation_results['overall_status'] = '❌'
        finally:
            cursor.close()
        
        return validation_results
    
    def print_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Print a formatted validation report"""
        print("\n" + "="*60)
        print("DATABASE SETUP VALIDATION REPORT")
        print("="*60)
        
        for category, details in validation_results.items():
            if category == 'overall_status':
                continue
                
            print(f"\n{category.upper().replace('_', ' ')}: {details.get('status', '❓')}")
            
            if 'required' in details and 'found' in details:
                print(f"  Required: {details['required']}")
                print(f"  Found: {details['found']}")
                if details.get('missing'):
                    print(f"  Missing: {details['missing']}")
            elif 'expected_after_data_load' in details:
                print(f"  Expected after data load: {details['expected_after_data_load']}")
                print(f"  Currently found: {details['currently_found']}")
                print(f"  Note: {details['note']}")
            elif 'found' in details:
                print(f"  Found: {details['found']}")
        
        print(f"\nOVERALL STATUS: {validation_results.get('overall_status', '❓')}")
        
        if validation_results.get('overall_status') == '✅':
            print("\n🎉 Database setup completed successfully!")
            print("All required schemas, tables, warehouses, and configurations are in place.")
        else:
            print("\n⚠️  Database setup has issues that need to be addressed.")
            print("Please review the missing components above.")
        
        print("="*60)

def main():
    """Main execution function"""
    setup = DatabaseSetup()
    
    try:
        # Connect to Snowflake
        setup.connect()
        
        # Execute setup SQL
        sql_file = Path(__file__).parent.parent.parent / "snowflake_sql" / "complete_database_setup.sql"
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL setup file not found: {sql_file}")
        
        logger.info("Starting database setup...")
        results = setup.execute_sql_file(sql_file)
        logger.info("Database setup SQL execution completed")
        
        # Validate setup
        validation_results = setup.validate_setup()
        
        # Print report
        setup.print_validation_report(validation_results)
        
        # Log summary
        if validation_results.get('overall_status') == '✅':
            logger.info("Database setup validation: SUCCESS")
            return 0
        else:
            logger.error("Database setup validation: FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return 1
    finally:
        setup.disconnect()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)