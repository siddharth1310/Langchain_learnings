# logger_setup.py
import logging
import os
from pythonjsonlogger import jsonlogger
from colorlog import ColoredFormatter


# Directory where all logs will be stored
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(file_name: str):
    """
    Creates and configures a logger for the given file.
    - Writes logs as JSONL in 'logs/<file>.jsonl'
    - Shows colorized logs in the terminal
    """

    base_name = os.path.basename(file_name).replace(".py", "")
    log_file = os.path.join(LOG_DIR, f"{base_name}.jsonl")

    # Create or get logger instance for the file
    logger = logging.getLogger(base_name)
    logger.setLevel(logging.DEBUG)  # Use INFO in production if too verbose

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # 1️⃣ File Handler → JSONL structured logs
    file_handler = logging.FileHandler(log_file)
    json_formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(message)s %(filename)s %(funcName)s %(lineno)d',
        datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
    )
    file_handler.setFormatter(json_formatter)

    # 2️⃣ Console Handler → Colored, human-readable logs
    console_handler = logging.StreamHandler()
    color_formatter = ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d - %(funcName)s()]%(reset)s %(message_log_color)s%(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
        secondary_log_colors={
            'message': {
                'ERROR': 'red',
                'WARNING': 'yellow',
                'INFO': 'white',
                'DEBUG': 'cyan',
                'CRITICAL': 'bold_red'
            }
        },
        style='%'
    )
    console_handler.setFormatter(color_formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
