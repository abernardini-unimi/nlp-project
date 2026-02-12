import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import os

from config.settings import LOGGER_DIRECTORY, LOGGER_LEVEL

# Create logs folder if it doesn't exist
os.makedirs(LOGGER_DIRECTORY, exist_ok=True)

class AlignedFormatter(logging.Formatter):
    def format(self, record):
        record.custom_file_line = f"{record.filename}:{record.lineno}"
        return super().format(record)

# Logger name with today's date (format YYYYMMDD)
logger_name = datetime.now().strftime("%Y%m%d")
logger = logging.getLogger(logger_name)
logger.setLevel(LOGGER_LEVEL)

# Formatter with custom alignment
file_formatter = AlignedFormatter(
    "%(levelname)s - %(message)s"
)

# File handler for the main logger
main_file_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8"
)
main_file_handler.setLevel(LOGGER_LEVEL)
main_file_handler.setFormatter(file_formatter)
main_file_handler.suffix = "%Y-%m-%d"

# 🔹 Console handler (to also print logs to the terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGER_LEVEL)
console_handler.setFormatter(file_formatter)

# Add both handlers to the main logger
logger.addHandler(main_file_handler)
logger.addHandler(console_handler)

# Disable propagation to the root logger to avoid duplicate logs
logger.propagate = False
