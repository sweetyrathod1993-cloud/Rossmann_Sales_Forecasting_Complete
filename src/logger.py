"""
Centralized logging configuration for Rossmann Store Sales Forecasting Project.
Adheres to Task 1.2 requirement: "Log your steps using the logger library in python."
"""

import os
import sys
import logging
from pathlib import Path

# Silence optional legacy numexpr/bottleneck incompatibilities if present
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sales_project.log"

_is_configured = False


def setup_logger(name: str = "rossmann_sales", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance writing to both console and file.
    """
    global _is_configured
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not _is_configured and not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _is_configured = True

    return logger


def get_logger(name: str = "rossmann_sales") -> logging.Logger:
    """Convenience getter for existing or newly initialized logger."""
    return setup_logger(name)


if __name__ == "__main__":
    test_logger = get_logger("test_logger")
    test_logger.info("Centralized logger initialized successfully.")
    test_logger.debug("Debug message test.")
    test_logger.warning("Warning message test.")
    print(f"Log file successfully created at: {LOG_FILE}")
