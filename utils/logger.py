import os
import logging
from logging import Logger as BaseLogger
from logging.handlers import RotatingFileHandler
from enum import Enum
from colorama import init, Fore, Style

# Initialize colorama for colored output on all platforms
init(autoreset=True)


class LoggerLevel(Enum):
    TRACE    = (5,  Fore.WHITE)
    DEBUG    = (10, Fore.CYAN)
    INFO     = (20, Fore.GREEN)
    WARNING  = (30, Fore.YELLOW)
    ERROR    = (40, Fore.RED)
    CRITICAL = (50, Fore.RED + Style.BRIGHT)

    @property
    def level_no(self):
        return self.value[0]

    @property
    def color(self):
        return self.value[1]


class ColoredFormatter(logging.Formatter):
    """
    Formatter that applies Colorama colors based on the record's level.
    """
    def format(self, record):
        # First get the standard formatted line
        formatted_line = super().format(record)
        # Pick the color for this level
        color = Fore.WHITE
        for lvl in LoggerLevel:
            if record.levelno == lvl.level_no:
                color = lvl.color
                break
        # Return the entire line wrapped in color codes
        return f"{color}{formatted_line}{Style.RESET_ALL}"


class Logger:
    """
    Wrapper around Python's logging.Logger that:
      - adds one colored console handler (singleton)
    """
    DEFAULT_LOG_DIR      = "logs"
    CORE_LOGFILE         = os.path.join(DEFAULT_LOG_DIR, "core.log")
    _instances           = {}
    _console_handlers     = []
    _file_handlers = []

    @classmethod
    def set_level(cls, lvl: LoggerLevel):
        """
        Set global logging level on root logger and console handler.
        """
        root = logging.getLogger()
        root.setLevel(lvl.level_no)

        for ch in cls._console_handlers:
            ch.setLevel(lvl.level_no)

        for fh in cls._file_handlers:
            fh.setLevel(lvl.level_no)

    @classmethod
    def get_logger(cls, clazz: type) -> BaseLogger:
        """
        Get a configured Logger instance for the given class.

        clazz: the class object
        """
        # Derive module_name as "package.module.ClassName"
        module_name = f"{clazz.__module__}.{clazz.__name__}"
        key = module_name
        if key in cls._instances:
            return cls._instances[key]

        # Create new logger
        logger = logging.getLogger(module_name)

        # Configure handlers only once per logger
        if not logger.handlers:
            # --- Console handler singleton ---
            ch = logging.StreamHandler()
            fmt = "%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
            ch.setFormatter(ColoredFormatter(fmt, "%Y-%m-%d %H:%M:%S"))
            ch.setLevel(logging.getLogger().level)
            logger.addHandler(ch)
            cls._console_handlers.append(ch)


            log_path = cls.CORE_LOGFILE

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = RotatingFileHandler(
                log_path,
                maxBytes=5_242_880,
                backupCount=3,
                encoding="utf-8"
            )
            fh.setLevel(logging.getLogger().level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s - [%(levelname)s] %(name)s - %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)
            cls._file_handlers.append(fh)

        # Inherit global level
        logger.setLevel(logging.DEBUG)
        cls._instances[key] = logger
        return logger