"""dbt resource configuration for Dagster."""

from pathlib import Path

from dagster_dbt import DbtCliResource, DbtProject

from dagster_project.constants import DBT_PROJECT_DIR

# dbt profiles.yml lives in the user's home directory (~/.dbt/)
DBT_PROFILES_DIR = Path.home() / ".dbt"

# DbtProject wraps the dbt project directory and locates the manifest.
# prepare_if_dev() auto-generates the manifest when running `dagster dev`.
dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR)
dbt_project.prepare_if_dev()

# The DbtCliResource is passed to @dbt_assets and used to invoke dbt CLI commands.
dbt_resource = DbtCliResource(
    project_dir=dbt_project,
    profiles_dir=DBT_PROFILES_DIR,
)
