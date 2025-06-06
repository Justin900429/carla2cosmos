import logging
import pathlib


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(out_dir: pathlib.Path, level: str):
    log_path = out_dir / "record.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Logging initialised – file at %s", log_path)
    logger = logging.getLogger("recorder")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)
    return logger
