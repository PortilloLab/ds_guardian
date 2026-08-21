import logging
import sys

# Códigos de color ANSI para consola interactiva
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

def get_logger(name: str = "ds_guardian") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ANSIFormatter('%(message)s'))
        logger.addHandler(handler)
    return logger
