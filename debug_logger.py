"""
debug_logger.py

Central debug logger for EvoSQL-Lite.
All modules import from here so log format is consistent.

Output goes to:
  - Terminal (always)
  - evosql_debug.log (always, rotates each run)
"""

import logging
import sys
from datetime import datetime

# ── Formatter ──────────────────────────────────────────────
class EvoFormatter(logging.Formatter):
    LEVEL_TAGS = {
        logging.DEBUG:    "🔍 DEBUG",
        logging.INFO:     "ℹ️  INFO ",
        logging.WARNING:  "⚠️  WARN ",
        logging.ERROR:    "❌ ERROR",
        logging.CRITICAL: "🔥 CRIT ",
    }

    def format(self, record):
        tag = self.LEVEL_TAGS.get(record.levelno, "     ")
        ts  = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        return f"[{ts}] {tag} | {record.name:<20} | {record.getMessage()}"


LOG_FILE = "evosql_debug.log"


def reset_log() -> None:
    """Call once per query run from app.py — clears the log file and rewrites fresh header."""
    # Close and remove all existing file handlers
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    # Overwrite with fresh header
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  EvoSQL-Lite Debug Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n\n")
    # Re-attach fresh file handlers
    fmt = EvoFormatter()
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Call this at the top of every module: from debug_logger import get_logger"""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = EvoFormatter()

    # Terminal handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger