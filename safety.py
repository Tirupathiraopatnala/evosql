"""
safety.py – now filters extracted schemas against known database schemas (case‑insensitive)
"""

import re
from typing import Tuple, List

from debug_logger import get_logger

log = get_logger("safety")


class SafetyGovernor:

    FORBIDDEN_KEYWORDS = [
        "DROP", "DELETE", "UPDATE", "TRUNCATE", "ALTER",
        "MERGE", "INSERT", "GRANT", "REVOKE", "CREATE",
        "EXEC", "EXECUTE"
    ]

    def validate(self, sql: str, required_schemas: List[str] = None) -> Tuple[bool, str]:
        log.info("SafetyGovernor.validate() called")
        log.debug(f"Required schemas: {required_schemas}")

        if not sql:
            log.error("Empty SQL — rejected")
            return False, "EMPTY_SQL"

        normalized = sql.upper().strip()

        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            first_word = normalized.split()[0] if normalized.split() else "?"
            log.warning(f"SQL does not start with SELECT/WITH — starts with '{first_word}'")
            return False, "NON_SELECT_STATEMENT"

        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", normalized):
                log.warning(f"Forbidden keyword found: {keyword}")
                return False, f"FORBIDDEN_OPERATION_{keyword}"

        if required_schemas:
            log.debug(f"Running schema validation — checking for: {required_schemas}")
            is_valid, reason = self._validate_schemas(sql, required_schemas)
            if not is_valid:
                log.warning(f"Schema validation FAILED: {reason}")
                return False, reason
            log.debug("Schema validation passed")

        log.info("SafetyGovernor.validate() → SAFE ✅")
        return True, "SAFE"

    # --------------------------------------------------
    def _validate_schemas(self, sql: str, required_schemas: List[str]) -> Tuple[bool, str]:
        sql_upper = sql.upper()
        missing = []

        for schema in required_schemas:
            schema_upper = schema.upper()
            if schema_upper not in sql_upper:
                missing.append(schema)
                log.debug(f"  Schema '{schema}' → MISSING from rewritten SQL")
            else:
                log.debug(f"  Schema '{schema}' → present ✅")

        if missing:
            return False, f"SCHEMA_REMOVED: {', '.join(missing)}"
        return True, "SCHEMAS_VALID"

    # --------------------------------------------------
    @staticmethod
    def extract_schemas(sql: str, known_schemas: List[str] = None) -> List[str]:
        """
        Extract schema names from original query.
        If known_schemas is provided, only return those that appear in the list
        (case‑insensitive matching).
        """
        log.info("SafetyGovernor.extract_schemas() called")

        # SQL keywords to ignore
        SQL_KEYWORDS = {
            "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR",
            "GROUP", "ORDER", "BY", "AS", "IN", "NOT", "NULL",
            "INNER", "LEFT", "RIGHT", "OUTER", "WITH", "HAVING",
            "UNION", "ALL", "DISTINCT", "TOP", "SET", "INTO"
        }

        # Build a set of uppercase known schemas for case‑insensitive matching
        if known_schemas is not None:
            known_set = {s.upper() for s in known_schemas}
        else:
            known_set = None

        # Match anything that looks like schema.table
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*'
        raw_matches = re.findall(pattern, sql)
        log.debug(f"Raw regex matches (before filtering): {raw_matches}")

        schemas = []
        for match in raw_matches:
            # Skip single characters (likely aliases)
            if len(match) <= 1:
                log.debug(f"  Skipping '{match}' — single character (likely alias)")
                continue
            # Skip SQL keywords
            if match.upper() in SQL_KEYWORDS:
                log.debug(f"  Skipping '{match}' — SQL keyword")
                continue
            # If we have a list of known schemas, check membership case‑insensitively
            if known_set is not None and match.upper() not in known_set:
                log.debug(f"  Skipping '{match}' — not a known schema")
                continue
            schemas.append(match)

        unique_schemas = list(set(schemas))
        log.info(f"extract_schemas() → {unique_schemas}")
        return unique_schemas