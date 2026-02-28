"""
synapse_client.py  — v3
Keeps original EXPLAIN + weighted fitness approach (gives non-zero signal).
Adds:
  - proxy_execute()  — TOP N execution for quick agent ranking (optional use)
  - strip_top_n()    — removes any TOP N the LLM may have copied from context
  - compute_checksum() fixed for surrogate char crash (pyodbc nvarchar/XML columns)
"""

import pyodbc
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, Optional
from contextlib import contextmanager


class SynapseClient:

    def __init__(self, server: str, database: str, username: str, password: str):
        self.connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = pyodbc.connect(self.connection_string, autocommit=True)
            yield conn
        finally:
            if conn:
                conn.close()

    # --------------------------------------------------
    # Execute Full Query
    # --------------------------------------------------
    def execute_query(self, sql: str, timeout: int = 300) -> Tuple[list, Dict[str, Any]]:
        metrics = {"execution_time": 0, "row_count": 0}
        start_time = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            conn.timeout = timeout
            cursor.execute(sql)
            rows = cursor.fetchall()
            metrics["execution_time"] = time.time() - start_time
            metrics["row_count"] = len(rows)
        return rows, metrics

    # --------------------------------------------------
    # Proxy Execute  (optional — TOP N for quick ranking)
    # --------------------------------------------------
    def proxy_execute(self, sql: str, top_n: int, timeout: int = 60) -> Dict[str, Any]:
        """
        Injects TOP N into the outermost SELECT and executes.
        Useful for quick ranking when query shape allows early exit.
        NOTE: Does not short-circuit for CTE-based queries — use with awareness.
        """
        proxied_sql = self._inject_top_n(sql, top_n)
        metrics = {"execution_time": 0, "row_count": 0, "proxy_n": top_n}
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                conn.timeout = timeout
                cursor.execute(proxied_sql)
                rows = cursor.fetchall()
                metrics["execution_time"] = time.time() - start_time
                metrics["row_count"] = len(rows)
        except Exception as e:
            metrics["execution_time"] = float(timeout)
            metrics["error"] = str(e)
            print(f"[SynapseClient] proxy_execute FAILED: {e}")
        return metrics

    # --------------------------------------------------
    # TOP N injection / stripping
    # --------------------------------------------------
    @staticmethod
    def _inject_top_n(sql: str, top_n: int) -> str:
        replaced = re.sub(r'(?i)(SELECT)\s+TOP\s+\(\d+\)\s+', f'SELECT TOP ({top_n}) ', sql, count=1)
        replaced = re.sub(r'(?i)(SELECT)\s+TOP\s+\d+\s+', f'SELECT TOP ({top_n}) ', replaced, count=1)
        if replaced == sql:
            replaced = re.sub(r'(?i)\bSELECT\b', f'SELECT TOP ({top_n})', sql, count=1)
        return replaced

    @staticmethod
    def strip_top_n(sql: str) -> str:
        """Remove any TOP N clause. Prevents LLM-copied proxy limits polluting stored rewrites."""
        cleaned = re.sub(r'(?i)\bSELECT\s+TOP\s+\(\d+\)\s*', 'SELECT ', sql)
        cleaned = re.sub(r'(?i)\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', cleaned)
        return cleaned

    # --------------------------------------------------
    # EXPLAIN Query via SET SHOWPLAN_XML ON
    # --------------------------------------------------
    def explain_query(self, sql: str) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SET SHOWPLAN_XML ON")
            cursor.nextset()
            cursor.execute(sql)
            plan_rows = cursor.fetchall()
            cursor.nextset()
            cursor.execute("SET SHOWPLAN_XML OFF")
        plan_xml = "\n".join(str(r[0]) for r in plan_rows if r[0])
        return self._parse_showplan_xml(plan_xml)

    # --------------------------------------------------
    # Parse ShowPlan XML
    # --------------------------------------------------
    def _parse_showplan_xml(self, plan_xml: str) -> Dict[str, Any]:
        metrics = {
            "total_cost":    0.0,
            "estimate_rows": 0,
            "shuffle_ops":   0,
            "broadcast_ops": 0,
            "full_scans":    0,
            "stale_stats":   0
        }

        if not plan_xml:
            return metrics

        try:
            plan_xml_clean = plan_xml.replace(
                ' xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan"', ''
            )
            root = ET.fromstring(plan_xml_clean)

            # Sum cost across all RelOp nodes.
            # Double-counts subtrees but gives a non-zero differential signal —
            # more useful than root-only which returns 0 on Synapse dedicated pools.
            total_cost = 0.0
            for relop in root.findall('.//RelOp'):
                cost_str = relop.get('EstimatedTotalSubtreeCost') or relop.get('PDWAccumulativeCost')
                if cost_str:
                    try:
                        total_cost += float(cost_str)
                    except ValueError:
                        pass
            metrics["total_cost"] = total_cost

            root_relop = root.find('.//RelOp')
            if root_relop is not None:
                metrics["estimate_rows"] = float(root_relop.get("EstimateRows", 0) or 0)

            for move in root.findall('.//Move'):
                move_type = move.get("MoveType", "")
                if move_type == "Shuffle":
                    metrics["shuffle_ops"] += 1
                elif move_type == "Broadcast":
                    metrics["broadcast_ops"] += 1

            for get in root.findall('.//Get'):
                if get.get("IsRoundRobin") == "true":
                    metrics["full_scans"] += 1

            metrics["stale_stats"] = len(
                root.findall('.//ColumnsWithStaleStatistics/ColumnReference')
            )

        except ET.ParseError as e:
            print(f"[SynapseClient] SHOWPLAN_XML parse error: {e}")

        return metrics

    # --------------------------------------------------
    # Compute Deterministic Checksum (order-independent)
    # --------------------------------------------------
    def compute_checksum(self, rows: list) -> str:
        """
        Surrogate-safe checksum — handles pyodbc column types (XML, binary,
        nvarchar with special chars) that produce non-JSON-serializable strings.
        """
        def _safe_str(col) -> str:
            try:
                s = "NULL" if col is None else str(col)
                return s.encode("utf-8", errors="replace").decode("utf-8")
            except Exception:
                return repr(col)

        serialized = sorted(
            "|".join(_safe_str(col) for col in row)
            for row in rows
        )
        hasher = hashlib.sha256()
        for row_str in serialized:
            hasher.update(row_str.encode("utf-8", errors="replace"))
        return hasher.hexdigest()

    # --------------------------------------------------
    # Safe Execution Wrapper
    # --------------------------------------------------
    def safe_execute(self, sql: str, timeout: int = 300) -> Optional[Tuple[list, Dict]]:
        try:
            return self.execute_query(sql, timeout)
        except Exception as e:
            print(f"[Synapse ERROR]: {e}")
            return None
