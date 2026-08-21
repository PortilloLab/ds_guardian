import logging
import sys
import json
from datetime import datetime

C_GREEN = '[92m'
C_RED = '[91m'
C_YELLOW = '[93m'
C_BLUE = '[94m'
C_BOLD = '[1m'
C_RESET = '[0m'

class ANSIFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            prefix = f"{C_BOLD}{C_RED}❌ ERROR: {C_RESET}"
        elif record.levelno == logging.WARNING:
            prefix = f"{C_BOLD}{C_YELLOW}⚠️ WARN: {C_RESET}"
        elif record.levelno == logging.INFO:
            prefix = f"{C_GREEN}ℹ️ {C_RESET}"
        else:
            prefix = ""
        return f"{prefix}{super().format(record)}"

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        return json.dumps(log_obj)

def get_logger(name: str = "ds_guardian", json_format: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        if json_format:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(ANSIFormatter('%(message)s'))
        logger.addHandler(handler)
    return logger
