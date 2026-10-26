import logging
import sys
from datetime import datetime
from pathlib import Path


def configure_eda_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    """Log EDA output to both the terminal and a timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"03_eda_{timestamp}.log"

    logger = logging.getLogger("diurnal_eda")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("Log file: %s", log_file)
    return logger, log_file
