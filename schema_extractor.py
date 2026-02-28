"""
schema_extractor.py

Purpose:
---------
Extracts schema intelligence from Azure Synapse.

Responsibilities:
------------------
1. Extract table distribution strategies
2. Extract index metadata
3. Extract partition metadata
4. Build structured metadata dictionary
5. Provide prompt-ready schema summary

This enables schema-aware evolutionary rewriting.
"""

from typing import Dict, Any


class SchemaExtractor:

    def __init__(self, synapse_client):
        self.synapse = synapse_client

    # --------------------------------------------------
    # Extract Distribution Strategy
    # --------------------------------------------------
    def get_distributions(self) -> Dict[str, str]:

        query = """
        SELECT
            t.name AS table_name,
            dp.distribution_policy_desc
        FROM sys.pdw_table_distribution_properties dp
        JOIN sys.tables t ON dp.object_id = t.object_id
        """

        rows, _ = self.synapse.execute_query(query)

        distributions = {}

        for row in rows:
            table_name = row[0]
            distribution_type = row[1]
            distributions[table_name] = distribution_type

        return distributions

    # --------------------------------------------------
    # Extract Index Metadata
    # --------------------------------------------------
    def get_indexes(self) -> Dict[str, list]:

        query = """
        SELECT
            t.name AS table_name,
            i.name AS index_name,
            i.type_desc
        FROM sys.indexes i
        JOIN sys.tables t ON i.object_id = t.object_id
        WHERE i.name IS NOT NULL
        """

        rows, _ = self.synapse.execute_query(query)

        index_map = {}

        for row in rows:
            table_name = row[0]
            index_name = row[1]
            index_type = row[2]

            if table_name not in index_map:
                index_map[table_name] = []

            index_map[table_name].append(f"{index_name} ({index_type})")

        return index_map

    # --------------------------------------------------
    # Extract Partition Metadata
    # --------------------------------------------------
    def get_partitions(self) -> Dict[str, Any]:

        query = """
        SELECT
            t.name AS table_name,
            p.partition_number
        FROM sys.partitions p
        JOIN sys.tables t ON p.object_id = t.object_id
        """

        rows, _ = self.synapse.execute_query(query)

        partition_map = {}

        for row in rows:
            table_name = row[0]

            if table_name not in partition_map:
                partition_map[table_name] = 0

            partition_map[table_name] += 1

        return partition_map

    # --------------------------------------------------
    # Build Unified Metadata
    # --------------------------------------------------
    def extract_schema_metadata(self) -> Dict[str, Any]:

        distributions = self.get_distributions()
        indexes = self.get_indexes()
        partitions = self.get_partitions()

        metadata = {}

        all_tables = set(distributions.keys()) | set(indexes.keys()) | set(partitions.keys())

        for table in all_tables:
            metadata[table] = {
                "distribution": distributions.get(table, "UNKNOWN"),
                "indexes": indexes.get(table, []),
                "partition_count": partitions.get(table, 0)
            }

        return metadata

    # --------------------------------------------------
    # Extract table names referenced in a SQL query
    # --------------------------------------------------
    @staticmethod
    def extract_tables_from_sql(sql: str) -> list:
        """
        Extract only the table names actually used in the submitted SQL.
        - Strips string literals first to avoid matching values inside quotes
        - Captures the table part of schema.table patterns
        - Captures bare table names after FROM/JOIN (skipping schemas followed by a dot)
        """
        import re

        # Remove string literals so values like 'OPS_4_DMT.OPS_FACT_...' are not matched
        sql_clean = re.sub(r"'[^']*'", "''", sql)

        tables = set()

        # schema.table  →  capture only the TABLE part (after the dot)
        for match in re.finditer(
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\.([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_clean
        ):
            tables.add(match.group(1).upper())

        # Bare table name after FROM/JOIN — skip if followed by a dot (it's a schema, not a table)
        for match in re.finditer(
            r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*\.)\b',
            sql_clean, re.IGNORECASE
        ):
            name = match.group(1).upper()
            if name not in {"SELECT", "WITH", "WHERE", "ON", "AS", "SET", "LATERAL"}:
                tables.add(name)

        return list(tables)

    # --------------------------------------------------
    # Convert Metadata to Prompt Text (query-specific)
    # --------------------------------------------------
    def build_prompt_metadata(self, sql: str = None) -> str:
        """
        Build schema metadata prompt scoped to tables in the submitted query.
        If sql is provided, only the tables referenced in that query are included —
        avoiding dumping thousands of irrelevant tables into the LLM prompt.
        Falls back to all tables if sql is not provided or no matches found.
        """
        metadata = self.extract_schema_metadata()

        if sql:
            referenced = self.extract_tables_from_sql(sql)
            metadata_upper = {k.upper(): v for k, v in metadata.items()}
            filtered = {
                table: metadata_upper[table]
                for table in referenced
                if table in metadata_upper
            }
            if filtered:
                note = f"Schema Metadata ({len(filtered)} table(s) referenced in your query):\n"
            else:
                # No matches — fall back to full metadata
                filtered = metadata
                note = "Schema Metadata (all tables — query tables not found in sys.tables):\n"
        else:
            filtered = metadata
            note = "Schema Metadata (all tables):\n"

        prompt_text = note
        for table, details in filtered.items():
            prompt_text += f"\nTable: {table}\n"
            prompt_text += f"  Distribution: {details['distribution']}\n"
            prompt_text += f"  Partition Count: {details['partition_count']}\n"
            if details["indexes"]:
                prompt_text += "  Indexes:\n"
                for idx in details["indexes"]:
                    prompt_text += f"    - {idx}\n"

        return prompt_text