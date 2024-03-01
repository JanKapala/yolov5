"""Project logging configuration."""

import logging
from typing import Optional

_LOGGER = "LOGGER"
LOGGER = logging.getLogger(_LOGGER)


def init_logging(
    log_file_path: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the process it is executed in.

    Args:
        level: Min log level at which and above performance_logs will be
            visible.
        format_string: Format string.
    """
    format_string = (
        format_string or "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(_LOGGER)
    logger.setLevel(level)

    for handler in list(logger.handlers):
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path, mode="a", delay=True)

    # Set levels for handlers
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    formatter = logging.Formatter(format_string)

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logging.basicConfig(level=level, handlers=[console_handler, file_handler])
    logger.propagate = False
