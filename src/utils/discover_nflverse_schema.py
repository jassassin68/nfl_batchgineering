#!/usr/bin/env python3
"""
Discover actual nflverse data schemas
Downloads sample data to understand the real column structure
"""

import requests
import polars as pl
from io import BytesIO
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def discover_schema(dataset_name: str, url: str) -> Dict:
    """Download and analyze schema of a nflverse dataset"""
    print(f"\n🔍 Discovering schema for {dataset_name}")
    print(f"📥 URL: {url}")
    
    try:
        # Download the parquet file
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Read into polars DataFrame
        df = pl.read_parquet(BytesIO(response.content))
        
        # Get schema information
        schema_info = {
            'dataset': dataset_name,
            'url': url,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': []
        }
        
        # Analyze each column
        for col_name in df.columns:
            col_info = {
                'name': col_name,
                'polars_type': str(df[col_name].dtype),
                'null_count': df[col_name].null_count(),
                'sample_values': []
            }
            
            # Get sample non-null values
            non_null_values = df[col_name].drop_nulls().head(3).to_list()
            col_info['sample_values'] = [str(v) for v in non_null_values]
            
            schema_info['columns'].append(col_info)
        
        print(f"✅ {dataset_name}: {len(df):,} rows, {len(df.columns)} columns")
        
        # Show first few columns as preview
        print("📋 First 10 columns:")
        for i, col in enumerate(schema_info['columns'][:10]):
            sample_str = ', '.join(col['sample_values'][:2]) if col['sample_values'] else 'NULL'
            print(f"  {i+1:2d}. {col['name']:<25} ({col['polars_type']:<12}) - e.g., {sample_str}")
        
        if len(schema_info['columns']) > 10:
            print(f"  ... and {len(schema_info['columns']) - 10} more columns")
        
        return schema_info
        
    except Exception as e:
        print(f"❌ Failed to discover schema for {dataset_name}: {e}")
        return {
            'dataset': dataset_name,
            'url': url,
            'error': str(e)
        }

def main():
    """Discover schemas for all nflverse datasets"""
    print("🏈 NFL Verse Data Schema Discovery")
    print("=" * 50)
    
    # Dataset URLs from TrainingDataLoader
    datasets = {
        'play_by_play': 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet',
        'rosters': 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2024.parquet',
        'team_summary_stats': 'https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_regpost_2024.parquet',
        'player_summary_stats': 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_regpost_2024.parquet',
        'play_by_play_participation': 'https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_2024.parquet',
        'injuries': 'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2024.parquet'
    }
    
    schemas = {}
    
    for dataset_name, url in datasets.items():
        schemas[dataset_name] = discover_schema(dataset_name, url)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 SCHEMA DISCOVERY SUMMARY")
    print(f"{'='*50}")
    
    successful = 0
    total_columns = 0
    
    for dataset_name, schema in schemas.items():
        if 'error' in schema:
            print(f"❌ {dataset_name}: {schema['error']}")
        else:
            successful += 1
            total_columns += schema['total_columns']
            print(f"✅ {dataset_name}: {schema['total_rows']:,} rows, {schema['total_columns']} columns")
    
    print(f"\n📈 Successfully discovered {successful}/{len(datasets)} schemas")
    print(f"📊 Total columns across all datasets: {total_columns}")
    
    # Generate CREATE TABLE statements
    print(f"\n{'='*50}")
    print("🛠️  SUGGESTED TABLE STRUCTURES")
    print(f"{'='*50}")
    
    for dataset_name, schema in schemas.items():
        if 'error' not in schema:
            print(f"\n-- {dataset_name.upper()} ({schema['total_columns']} columns)")
            print(f"CREATE OR REPLACE TABLE {dataset_name} (")
            
            # All columns as VARCHAR for flexibility
            column_defs = []
            for col in schema['columns']:
                column_defs.append(f'    "{col["name"]}" VARCHAR')
            
            # Add metadata columns
            column_defs.extend([
                '    "source_year" VARCHAR',
                '    "load_type" VARCHAR', 
                '    "loaded_at" TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()'
            ])
            
            print(',\n'.join(column_defs))
            print(f");")
            print(f"-- Sample columns: {', '.join([col['name'] for col in schema['columns'][:5]])}")
    
    print(f"\n💡 All columns are defined as VARCHAR for maximum flexibility.")
    print(f"💡 The TrainingDataLoader will handle the actual data loading and type conversion.")
    
    return schemas

if __name__ == "__main__":
    try:
        schemas = main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Schema discovery interrupted by user")
    except Exception as e:
        print(f"\n❌ Schema discovery failed: {e}")
        sys.exit(1)