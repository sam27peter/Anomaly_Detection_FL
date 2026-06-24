from pathlib import Path
import logging


LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    exist_ok=True
)


def get_logger(
        logger_name: str,
        log_filename: str
):
    """
    Create and return a logger.
    """

    logger = logging.getLogger(
        logger_name
    )

    logger.setLevel(
        logging.INFO
    )

    if logger.handlers:
        return logger

    formatter = logging.Formatter(

        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"

    )

    file_handler = logging.FileHandler(

        LOG_DIR / log_filename

    )

    file_handler.setFormatter(
        formatter
    )

    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        stream_handler
    )

    return logger


# ==================================================
# GLOBAL LOGGERS
# ==================================================

project_logger = get_logger(
    "project",
    "project.log"
)

centralized_logger = get_logger(
    "centralized",
    "centralized.log"
)

federated_logger = get_logger(
    "federated",
    "federated.log"
)

dashboard_logger = get_logger(
    "dashboard",
    "dashboard.log"
)

# Backward compatibility
logger = project_logger