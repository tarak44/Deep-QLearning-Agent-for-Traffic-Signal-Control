import logging
from pathlib import Path

import typer
from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to use RichHandler.

    Args:
        level: Logging level to configure for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[typer],
                tracebacks_show_locals=False,
            )
        ],
    )


configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the global Rich logging setup.

    Args:
        name: Name of the logger (usually ``__name__``).

    Returns:
        The logger instance with the given name.
    """
    return logging.getLogger(name)


def add_file_handler(log_path: Path) -> None:
    """Attach a file handler to the root logger if not already present.

    Args:
        log_path: Path to the log file.
    """
    root_logger = logging.getLogger()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(file_handler)
