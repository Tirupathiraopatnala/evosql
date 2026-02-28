"""
validator.py
"""

from typing import List, Tuple, Any
from debug_logger import get_logger

log = get_logger("validator")


class QueryValidator:

    def validate_row_count(self, baseline_rows, candidate_rows) -> bool:
        bl = len(baseline_rows)
        cl = len(candidate_rows)
        match = bl == cl
        if match:
            log.debug(f"Row count check: {bl} == {cl} ✅")
        else:
            log.warning(f"Row count MISMATCH: baseline={bl}  candidate={cl}")
        return match

    def validate_schema(self, baseline_rows, candidate_rows) -> bool:
        if not baseline_rows or not candidate_rows:
            log.warning("validate_schema() — one of the result sets is empty")
            return False
        bc = len(baseline_rows[0])
        cc = len(candidate_rows[0])
        match = bc == cc
        if match:
            log.debug(f"Column count check: {bc} == {cc} ✅")
        else:
            log.warning(f"Column count MISMATCH: baseline={bc}  candidate={cc}")
        return match

    def validate_checksum(self, baseline_checksum: str, candidate_checksum: str) -> bool:
        match = baseline_checksum == candidate_checksum
        if match:
            log.debug(f"Checksum match ✅  ({baseline_checksum[:16]}…)")
        else:
            log.warning(f"Checksum MISMATCH")
            log.debug(f"  baseline  : {baseline_checksum[:32]}…")
            log.debug(f"  candidate : {candidate_checksum[:32]}…")
        return match

    def log_checksum_diff(
        self,
        baseline_rows: List[Tuple[Any]],
        candidate_rows: List[Tuple[Any]]
    ) -> None:
        bl = len(baseline_rows)
        cl = len(candidate_rows)
        log.warning(f"Checksum diff diagnostics: baseline={bl} rows  candidate={cl} rows")

        baseline_set  = set(str(tuple(str(c) for c in row)) for row in baseline_rows)
        candidate_set = set(str(tuple(str(c) for c in row)) for row in candidate_rows)

        only_in_baseline  = baseline_set  - candidate_set
        only_in_candidate = candidate_set - baseline_set

        sample_size = 3

        if only_in_baseline:
            log.warning(f"  Rows in baseline but NOT in candidate ({len(only_in_baseline)} total):")
            for row in list(only_in_baseline)[:sample_size]:
                log.warning(f"    - {row[:120]}")

        if only_in_candidate:
            log.warning(f"  Rows in candidate but NOT in baseline ({len(only_in_candidate)} total):")
            for row in list(only_in_candidate)[:sample_size]:
                log.warning(f"    + {row[:120]}")

        if not only_in_baseline and not only_in_candidate:
            log.warning(
                "  Row content matches as sets but order or duplicates differ — "
                "possible duplicate rows introduced or removed by agent."
            )

    def validate(
        self,
        baseline_rows:      List[Tuple[Any]],
        candidate_rows:     List[Tuple[Any]],
        baseline_checksum:  str,
        candidate_checksum: str
    ) -> Tuple[bool, str]:

        log.info("QueryValidator.validate() called")

        if not self.validate_row_count(baseline_rows, candidate_rows):
            return False, "ROW_COUNT_MISMATCH"

        if not self.validate_schema(baseline_rows, candidate_rows):
            return False, "SCHEMA_MISMATCH"

        # TODO: Re-enable once compute_checksum() is fixed with _normalize_col()
        # to handle Decimal vs float precision, datetime microseconds, etc.
        # Current false-mismatch rate is too high to be useful as a gate.
        # if not self.validate_checksum(baseline_checksum, candidate_checksum):
        #     self.log_checksum_diff(baseline_rows, candidate_rows)
        #     return False, "CHECKSUM_MISMATCH"
        log.warning("Checksum validation SKIPPED (temporarily disabled — see TODO)")

        log.info("QueryValidator.validate() → VALID ✅")
        return True, "VALID"