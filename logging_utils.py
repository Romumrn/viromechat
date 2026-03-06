import logging
import os
import re
from datetime import datetime

# ANSI color codes
C = {
    'RESET': '\033[0m',
    'BOLD': '\033[1m',
    'DIM': '\033[2m',
    'ITALIC': '\033[3m',
    'UNDERLINE': '\033[4m',
    'REVERSE': '\033[7m',
    'RED': '\033[31m', 'GREEN': '\033[32m', 'YELLOW': '\033[33m',
    'BLUE': '\033[34m', 'MAGENTA': '\033[35m', 'CYAN': '\033[36m', 'WHITE': '\033[37m',
    'BRIGHT_RED': '\033[91m', 'BRIGHT_GREEN': '\033[92m', 'BRIGHT_YELLOW': '\033[93m',
    'BRIGHT_BLUE': '\033[94m', 'BRIGHT_MAGENTA': '\033[95m', 'BRIGHT_CYAN': '\033[96m', 'BRIGHT_WHITE': '\033[97m',
    'BG_RED': '\033[41m', 'BG_GREEN': '\033[42m', 'BG_YELLOW': '\033[43m',
    'BG_BLUE': '\033[44m', 'BG_MAGENTA': '\033[45m',
}

LEVEL_COLORS = {
    'DEBUG': C['CYAN'], 'INFO': C['BRIGHT_GREEN'], 'WARNING': C['BRIGHT_YELLOW'],
    'ERROR': C['BRIGHT_RED'], 'CRITICAL': C['BG_RED'] + C['BOLD'] + C['WHITE'],
}

KEYWORD_COLORS = {
    'USER_QUERY': C['BOLD'] + C['BRIGHT_BLUE'] + C['UNDERLINE'],
    'CONFIG': C['DIM'] + C['CYAN'],
    'THINKING': C['DIM'] + C['WHITE'],
    'TOOL_CALL': C['BOLD'] + C['BRIGHT_MAGENTA'],
    'TOOL_OK': C['BOLD'] + C['BRIGHT_GREEN'],
    'TOOL_FAIL': C['BOLD'] + C['BRIGHT_RED'] + C['REVERSE'],
    'TOOL_CONTENT_TRUNCATED': C['YELLOW'] + C['ITALIC'],
    'RESULT': C['BOLD'] + C['BRIGHT_GREEN'] + C['BG_GREEN'] + C['BLUE'],
    'ERROR': C['BOLD'] + C['BG_RED'] + C['WHITE'],
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        original_levelname = record.levelname
        level_color = LEVEL_COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{C['RESET']}"
        
        msg = record.getMessage()
        
        for keyword, color in KEYWORD_COLORS.items():
            if keyword in msg:
                msg = msg.replace(keyword, f"{color}{keyword}{C['RESET']}")
        
        msg = re.sub(r'(#\d+)', f"{C['BRIGHT_YELLOW']}{C['BOLD']}\\1{C['RESET']}", msg)
        msg = re.sub(r"(args=\{[^}]+\})", f"{C['DIM']}\\1{C['RESET']}", msg)
        msg = re.sub(r'(shape=\([^)]+\))', f"{C['BRIGHT_CYAN']}\\1{C['RESET']}", msg)
        msg = re.sub(r'(https?://[^\s]+)', f"{C['BRIGHT_BLUE']}{C['UNDERLINE']}\\1{C['RESET']}", msg)
        
        if 'Error:' in msg:
            msg = msg.replace('Error:', f"{C['BRIGHT_RED']}{C['BOLD']}Error:{C['RESET']}")
        
        record.msg = msg
        record.args = ()
        result = super().format(record)
        record.levelname = original_levelname
        return result

class PlainFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg
        record.args = ()
        return super().format(record)

def setup_logger(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = f"{LOG_DIR}/agent_{datetime.now().strftime('%Y-%m')}.log"
    
    logger = logging.getLogger("virus_agent")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
