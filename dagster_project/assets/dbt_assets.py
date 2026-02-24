"""dbt assets: auto-generates one Dagster asset per dbt model from the manifest.

Source keys use the default dagster-dbt convention (["nfl_verse_raw", "<table>"]),
which matches the keys produced by the ingestion multi_asset. This means
Dagster automatically wires the ingestion -> staging dependency edges.
"""

from typing import Any, Mapping, Optional

from dagster_dbt import DagsterDbtTranslator, DbtCliResource, dbt_assets

from dagster_project.resources.dbt_resource import dbt_project

MANIFEST_PATH = dbt_project.manifest_path


class NflDbtTranslator(DagsterDbtTranslator):
    """Custom translator to assign group names based on dbt model layer."""

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> Optional[str]:
        """Assign group names based on dbt model path.

        - 1_staging/*      -> "staging"
        - 2_intermediate/* -> "intermediate"
        - 3_marts/*        -> "marts"
        """
        fqn = dbt_resource_props.get("fqn", [])

        for part in fqn:
            if part.startswith("1_staging"):
                return "staging"
            if part.startswith("2_intermediate"):
                return "intermediate"
            if part.startswith("3_marts"):
                return "marts"

        return super().get_group_name(dbt_resource_props)


@dbt_assets(
    manifest=MANIFEST_PATH,
    dagster_dbt_translator=NflDbtTranslator(),
)
def nfl_dbt_assets(context, dbt: DbtCliResource):
    """Run dbt build (run + test) for the selected models."""
    yield from dbt.cli(["build"], context=context).stream()
