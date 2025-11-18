import datetime
import logging
import os


def setup_logging(level=logging.INFO, log_dir: str | None = "_runlog") -> None:
    class CustomColorFormatter(logging.Formatter):
        ORANGE = "\033[33m"  # Orange/yellow in most terminals
        RED = "\033[91m"
        RESET = "\033[0m"

        def format(self, record):
            # Format the full line first
            line = super().format(record)

            # Colorize based on level
            if record.levelno == logging.WARNING:
                return f"{self.ORANGE}{line}{self.RESET}"
            elif record.levelno >= logging.ERROR:
                return f"{self.RED}{line}{self.RESET}"
            else:
                return line  # No color for INFO/DEBUG by default

    # Define formatter (used for both console and file)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler (with color)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        CustomColorFormatter(formatter._fmt, formatter.datefmt)
    )

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear previous handlers
    root_logger.addHandler(console_handler)

    if log_dir is not None:
        # Generate session-specific log filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"log_{timestamp}.txt")

        # File handler (no color)
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)


def _demo():
    setup_logging(log_dir=None)
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.info("Back to info.")


if __name__ == "__main__":
    _demo()
